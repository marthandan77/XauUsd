from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPS = 1e-12
Z80 = 1.2815515655446004
FEATURES = [
    "ema_spread",
    "price_fast_distance",
    "rsi_centered",
    "adx_scaled",
    "atr_pct",
    "vwap_distance",
    "range_position",
    "kc_momentum_atr",
    "realized_vol",
    "candle_return",
    "squeeze_on",
    "squeeze_fired",
]


def _series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _bars_per_day(interval: str) -> int:
    return {"15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}.get(str(interval), 96)


def _interval_minutes(interval: str) -> int:
    return {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(str(interval), 15)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def build_feature_frame(bars: pd.DataFrame, settings: dict) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    close = _series(bars, "close").replace(0, np.nan)
    open_ = _series(bars, "open").replace(0, np.nan)
    atr = _series(bars, "atr").replace(0, np.nan)
    window = max(
        int(settings.get("lookback_days", 7))
        * _bars_per_day(str(settings.get("price_interval", "15m"))),
        2,
    )
    high = _series(bars, "high").rolling(window, min_periods=min(window, 20)).max()
    low = _series(bars, "low").rolling(window, min_periods=min(window, 20)).min()
    price_range = (high - low).replace(0, np.nan)
    out = pd.DataFrame(index=bars.index)
    out["ema_spread"] = (_series(bars, "ema_fast") - _series(bars, "ema_slow")) / close
    out["price_fast_distance"] = (close - _series(bars, "ema_fast")) / close
    out["rsi_centered"] = (_series(bars, "rsi", 50) - 50.0) / 50.0
    out["adx_scaled"] = _series(bars, "adx") / 100.0
    out["atr_pct"] = atr / close
    out["vwap_distance"] = (close - _series(bars, "vwap")) / close
    out["range_position"] = ((close - low) / price_range) - 0.5
    out["kc_momentum_atr"] = _series(bars, "kc_momentum") / atr
    out["realized_vol"] = _series(bars, "realized_vol")
    out["candle_return"] = np.log(close / open_)
    out["squeeze_on"] = _series(bars, "squeeze_on").fillna(0).astype(float)
    out["squeeze_fired"] = _series(bars, "squeeze_fired").fillna(0).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def historical_rule_edge(bars: pd.DataFrame, features: pd.DataFrame, settings: dict) -> pd.Series:
    close, open_ = _series(bars, "close"), _series(bars, "open")
    fast, slow = _series(bars, "ema_fast"), _series(bars, "ema_slow")
    trend = _series(bars, "trend_sma")
    rsi, adx = _series(bars, "rsi", 50), _series(bars, "adx")
    momentum, previous = _series(bars, "kc_momentum"), _series(bars, "kc_momentum_prev")
    squeeze = _series(bars, "squeeze_on").fillna(0).astype(bool)
    fired = _series(bars, "squeeze_fired").fillna(0).astype(bool)
    bull = pd.Series(0.0, index=bars.index)
    bear = pd.Series(0.0, index=bars.index)
    adx_min, rsi_level = float(settings.get("adx_minimum", 20)), float(settings.get("rsi_level", 50))
    bull += ((fast > slow) & (close > fast) & (adx >= adx_min)).astype(float) * 25
    bear += ((fast < slow) & (close < fast) & (adx >= adx_min)).astype(float) * 25
    position = features["range_position"] + 0.5
    support, resistance = position <= 0.10, position >= 0.90
    bull += support.astype(float) * 20 + (~resistance).astype(float) * 5
    bear += resistance.astype(float) * 20 + (~support).astype(float) * 5
    bull += (close > open_).astype(float) * 10 + (rsi >= rsi_level).astype(float) * 5
    bear += (close < open_).astype(float) * 10 + (rsi <= rsi_level).astype(float) * 5
    up_break = fired & (momentum > 0) & (momentum > previous) & (close > trend)
    down_break = fired & (momentum < 0) & (momentum < previous) & (close < trend)
    bull += up_break.astype(float) * float(settings.get("kc_breakout_score", 25))
    bear += down_break.astype(float) * float(settings.get("kc_breakout_score", 25))
    bull += ((~fired) & (~squeeze) & (momentum > 0) & (close > trend)).astype(float) * float(
        settings.get("kc_momentum_score", 10)
    )
    bear += ((~fired) & (~squeeze) & (momentum < 0) & (close < trend)).astype(float) * float(
        settings.get("kc_momentum_score", 10)
    )
    return ((bull - bear) / 100.0).clip(-1.0, 1.0)


def _ridge(train_x: np.ndarray, train_y: np.ndarray, predict_x: np.ndarray, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    predict = np.column_stack([np.ones(len(predict_x)), predict_x])
    regularizer = np.eye(design.shape[1]) * max(float(penalty), 0.0)
    regularizer[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + regularizer) @ design.T @ train_y
    return predict @ beta, beta


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    other = np.where(np.isfinite(other), other, mean)
    return (train - mean) / std, (other - mean) / std


def _ic(prediction: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    if len(actual) < 5 or np.std(prediction) < EPS or np.std(actual) < EPS:
        return 0.0, -1.0
    value = float(np.clip(np.corrcoef(prediction, actual)[0, 1], -0.999999, 0.999999))
    lower = math.tanh(math.atanh(value) - 1.96 / math.sqrt(len(actual) - 3))
    return value, float(lower)


def _evaluate_horizon(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    rule_edge: pd.Series,
    horizon: int,
    settings: dict,
    current_rule_edge: float,
) -> dict | None:
    frame = features.copy()
    frame["rule"] = rule_edge
    frame["target"] = np.log(_series(bars, "close").shift(-horizon) / _series(bars, "close"))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    minimum = max(int(settings.get("quant_min_samples", 100)), 40)
    if len(frame) < minimum:
        return None
    split = min(max(int(len(frame) * float(settings.get("quant_train_fraction", 0.70))), 30), len(frame) - 20)
    if split <= 0 or len(frame) - split < 20:
        return None

    x, y = frame[FEATURES].to_numpy(float), frame["target"].to_numpy(float)
    rule = frame["rule"].to_numpy(float).reshape(-1, 1)
    train_x, val_x = _standardize(x[:split], x[split:])
    quant_val, _ = _ridge(train_x, y[:split], val_x, float(settings.get("quant_ridge_lambda", 5.0)))
    rule_val, _ = _ridge(rule[:split], y[:split], rule[split:], 0.01)
    quant_rmse = float(np.sqrt(np.mean((y[split:] - quant_val) ** 2)))
    rule_rmse = float(np.sqrt(np.mean((y[split:] - rule_val) ** 2)))
    weight = (1 / max(quant_rmse**2, EPS)) / (
        1 / max(quant_rmse**2, EPS) + 1 / max(rule_rmse**2, EPS)
    )
    combined = weight * quant_val + (1 - weight) * rule_val
    residual = y[split:] - combined
    combined_rmse = float(np.sqrt(np.mean(residual**2)))
    naive_rmse = float(np.sqrt(np.mean(y[split:] ** 2)))
    sigma = max(float(np.std(residual, ddof=1)), EPS)
    conformal = max(float(np.quantile(np.abs(residual), 0.80)), Z80 * sigma)
    information, lower = _ic(combined, y[split:])

    all_x, current_x = _standardize(x, features[FEATURES].iloc[[-1]].to_numpy(float))
    _, quant_beta = _ridge(all_x, y, all_x, float(settings.get("quant_ridge_lambda", 5.0)))
    _, rule_beta = _ridge(rule, y, rule, 0.01)
    mu_quant = float((np.column_stack([np.ones(1), current_x]) @ quant_beta).item())
    mu_rule = float((np.array([[1.0, current_rule_edge]]) @ rule_beta).item())
    mu = weight * mu_quant + (1 - weight) * mu_rule

    return {
        "horizon_bars": int(horizon),
        "samples": int(len(frame)),
        "validation_samples": int(len(y) - split),
        "quant_rmse": quant_rmse,
        "rule_rmse": rule_rmse,
        "combined_rmse": combined_rmse,
        "naive_rmse": naive_rmse,
        "skill": float(1 - combined_rmse / max(naive_rmse, EPS)),
        "directional_accuracy": float(np.mean(np.sign(combined) == np.sign(y[split:]))),
        "coverage_80": float(np.mean(np.abs(residual) <= conformal)),
        "information_coefficient": information,
        "ic_lower_95": lower,
        "quant_weight": float(weight),
        "expected_log_return": float(mu),
        "forecast_sigma": sigma,
        "conformal_half_width": conformal,
    }


def _half_life(diagnostics: list[dict]) -> float | None:
    points = [(d["horizon_bars"], abs(d["information_coefficient"])) for d in diagnostics if d["information_coefficient"] > 1e-6]
    if len(points) < 2:
        return None
    slope, _ = np.polyfit([p[0] for p in points], np.log([p[1] for p in points]), 1)
    return float((-1 / slope) * math.log(2)) if slope < 0 else None


def _barrier_probability(mu: float, sigma: float, upper: float, lower: float) -> float:
    if upper <= 0 or lower <= 0:
        return 0.0
    if abs(mu) < 1e-9:
        return float(lower / (upper + lower))
    variance = max(sigma**2, EPS)
    numerator = 1 - math.exp(float(np.clip(-2 * mu * lower / variance, -700, 700)))
    denominator = 1 - math.exp(float(np.clip(-2 * mu * (upper + lower) / variance, -700, 700)))
    return float(np.clip(numerator / denominator, 0, 1)) if abs(denominator) > EPS else float(lower / (upper + lower))


def _expectancy(action: str, plan: dict, price: float, atr: float, mu: float, sigma: float, settings: dict) -> dict:
    cost = max(float(settings.get("quant_cost_atr_fraction", 0.05)), 0) * max(atr, 0)
    if action == "BUY PLAN" and plan.get("tp1") is not None and plan.get("stop") is not None:
        reward, risk = max(float(plan["tp1"]) - price, 0), max(price - float(plan["stop"]), 0)
        probability = _barrier_probability(mu, sigma, math.log(float(plan["tp1"]) / price), math.log(price / float(plan["stop"])))
    elif action == "SELL PLAN" and plan.get("tp1") is not None and plan.get("stop") is not None:
        reward, risk = max(price - float(plan["tp1"]), 0), max(float(plan["stop"]) - price, 0)
        probability = _barrier_probability(-mu, sigma, math.log(price / float(plan["tp1"])), math.log(float(plan["stop"]) / price))
    else:
        return {"tp_before_sl_probability": 0.0, "expected_value": 0.0, "cost": cost}
    return {
        "tp_before_sl_probability": probability,
        "expected_value": probability * reward - (1 - probability) * risk - cost,
        "cost": cost,
    }


def build_unified_forecast(
    bars: pd.DataFrame,
    scores: dict,
    plan: dict,
    veto: dict,
    settings: dict,
    scanned_at: pd.Timestamp,
) -> dict:
    if bars is None or bars.empty:
        return {"ready": False, "mode": "insufficient_data", "reason": "No enriched bars are available."}
    features = build_feature_frame(bars, settings)
    rule_history = historical_rule_edge(bars, features, settings)
    current_rule = float(np.clip((float(scores.get("bull_score", 0)) - float(scores.get("bear_score", 0))) / 100, -1, 1))
    horizons = sorted({max(int(h), 1) for h in settings.get("quant_candidate_horizons", [1, 2, 3, 4, 6, 8])})
    diagnostics = [d for h in horizons if (d := _evaluate_horizon(bars, features, rule_history, h, settings, current_rule))]
    if not diagnostics:
        return {
            "ready": False,
            "mode": "insufficient_data",
            "reason": "Not enough historical bars for walk-forward calibration.",
            "candidate_diagnostics": [],
        }
    viable = [d for d in diagnostics if d["skill"] > 0 and d["ic_lower_95"] > 0]
    if viable:
        selected, mode, gate = max(viable, key=lambda d: d["horizon_bars"]), "validated", True
    else:
        default = max(int(settings.get("quant_default_horizon_bars", 4)), 1)
        selected, mode, gate = min(diagnostics, key=lambda d: abs(d["horizon_bars"] - default)), "calibration", False

    price, atr = float(bars["close"].iloc[-1]), float(bars["atr"].iloc[-1])
    mu, sigma, width = selected["expected_log_return"], selected["forecast_sigma"], selected["conformal_half_width"]
    p_up = float(np.clip(_normal_cdf(mu / max(sigma, EPS)), 0, 1))
    action = str(veto.get("final_action", "HOLD"))
    expectancy = _expectancy(action, plan, price, atr, mu, sigma, settings)
    reasons: list[str] = []
    threshold = float(settings.get("quant_min_probability", 0.60))
    if gate and action == "BUY PLAN":
        if mu <= 0: reasons.append("quant expected return is not positive")
        if p_up < threshold: reasons.append("probability-up threshold not met")
        if expectancy["expected_value"] <= 0: reasons.append("trade expected value is not positive")
    elif gate and action == "SELL PLAN":
        if mu >= 0: reasons.append("quant expected return is not negative")
        if 1 - p_up < threshold: reasons.append("probability-down threshold not met")
        if expectancy["expected_value"] <= 0: reasons.append("trade expected value is not positive")
    final_action = "HOLD" if reasons and action in {"BUY PLAN", "SELL PLAN"} else action
    interval = str(settings.get("price_interval", "15m"))
    expiry = scanned_at + pd.Timedelta(minutes=selected["horizon_bars"] * _interval_minutes(interval))
    return {
        "ready": True,
        "mode": mode,
        "quant_gate_active": gate,
        "rule_action": action,
        "final_action": final_action,
        "quant_reasons": reasons,
        "anchor_price": price,
        "expected_price": price * math.exp(mu),
        "lower_price": price * math.exp(mu - width),
        "upper_price": price * math.exp(mu + width),
        "expected_log_return": mu,
        "forecast_sigma": sigma,
        "probability_up": p_up,
        "probability_down": 1 - p_up,
        "horizon_bars": selected["horizon_bars"],
        "scanned_at": scanned_at,
        "price_interval": interval,
        "expiry_time": expiry,
        "half_life_bars": _half_life(diagnostics),
        "tp_before_sl_probability": expectancy["tp_before_sl_probability"],
        "expected_value": expectancy["expected_value"],
        "estimated_cost": expectancy["cost"],
        "selected_diagnostics": selected,
        "candidate_diagnostics": diagnostics,
    }


def scan_validity(
    quant: dict,
    current_price: float | None = None,
    current_regime: str | None = None,
    current_atr: float | None = None,
    now: pd.Timestamp | None = None,
) -> dict:
    if not quant or not quant.get("ready"):
        return {
            "status": "UNAVAILABLE",
            "progress": 0.0,
            "remaining_seconds": 0.0,
            "price_deviation_sigma": None,
            "reasons": ["quant forecast unavailable"],
        }

    current = now or pd.Timestamp.now(tz="UTC")
    current = current.tz_localize("UTC") if current.tzinfo is None else current
    scanned, expiry = pd.Timestamp(quant["scanned_at"]), pd.Timestamp(quant["expiry_time"])
    scanned = scanned.tz_localize("UTC") if scanned.tzinfo is None else scanned
    expiry = expiry.tz_localize("UTC") if expiry.tzinfo is None else expiry
    remaining = max((expiry - current).total_seconds(), 0)
    progress = float(np.clip((current - scanned).total_seconds() / max((expiry - scanned).total_seconds(), 1), 0, 1))

    anchor = float(quant["anchor_price"])
    price = float(current_price) if current_price is not None else anchor
    mu = float(quant["expected_log_return"])
    sigma = max(float(quant["forecast_sigma"]), EPS)
    path_fraction = max(progress, 1.0 / max(int(quant.get("horizon_bars", 1)), 1))
    expected_log_return = mu * progress
    expected_path_price = anchor * math.exp(expected_log_return)
    path_sigma = sigma * math.sqrt(path_fraction)
    deviation = abs(math.log(max(price, EPS) / max(expected_path_price, EPS))) / max(path_sigma, EPS)

    reasons: list[str] = []
    if current >= expiry:
        reasons.append("forecast horizon elapsed")
    if deviation > 1.5:
        reasons.append("price moved outside the 1.5-sigma path envelope")
    scan_regime = quant.get("scan_regime")
    if scan_regime and current_regime and str(current_regime) != str(scan_regime):
        reasons.append("market regime changed")
    scan_atr = quant.get("scan_atr")
    if scan_atr and current_atr and float(scan_atr) > 0 and float(current_atr) > 0:
        if abs(math.log(float(current_atr) / float(scan_atr))) > math.log(1.5):
            reasons.append("ATR changed by more than the 1.5x volatility limit")

    if reasons:
        status = "EXPIRED"
    elif progress >= 0.75 or deviation > 1.0:
        status = "WEAKENING"
    else:
        status = "VALID"
    return {
        "status": status,
        "progress": progress,
        "remaining_seconds": remaining,
        "price_deviation_sigma": float(deviation),
        "expected_path_price": float(expected_path_price),
        "current_price": float(price),
        "reasons": reasons,
    }
