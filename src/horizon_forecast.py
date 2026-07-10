from __future__ import annotations

import math

import numpy as np
import pandas as pd


HORIZON_CONFIG = {
    "5m": {"minutes": 5, "role": "Signal", "mode": "entry"},
    "1h": {"minutes": 60, "role": "Filter", "mode": "filter"},
    "4h": {"minutes": 240, "role": "Filter", "mode": "filter"},
    "Daily": {"minutes": 1440, "role": "Filter", "mode": "filter"},
}

INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

DEFAULT_MIN_EV_POINTS = {
    "5m": 0.50,
    "1h": 1.80,
    "4h": 4.00,
    "Daily": 8.00,
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _base_minutes(settings: dict) -> int:
    return INTERVAL_MINUTES.get(str(settings.get("price_interval", "5m")), 5)


def _bars_for_minutes(settings: dict, minutes: int, minimum: int = 1) -> int:
    base = max(_base_minutes(settings), 1)
    return max(int(round(float(minutes) / float(base))), minimum)


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_now: np.ndarray, alpha: float = 2.0) -> tuple[float, float, int]:
    """Fit a small ridge model with closed-form linear algebra and predict the latest row."""
    valid = np.isfinite(y_train) & np.isfinite(x_train).all(axis=1)
    x_train = x_train[valid]
    y_train = y_train[valid]

    if len(y_train) < max(120, x_train.shape[1] * 10):
        return 0.0, float(np.nan), int(len(y_train))

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    xs = (x_train - mean) / std
    x_now_s = (x_now - mean) / std

    design = np.column_stack([np.ones(len(xs)), xs])
    ridge = np.eye(design.shape[1]) * float(alpha)
    ridge[0, 0] = 0.0
    try:
        beta = np.linalg.solve(design.T @ design + ridge, design.T @ y_train)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(design.T @ design + ridge) @ design.T @ y_train

    pred = float(np.r_[1.0, x_now_s] @ beta)
    residuals = y_train - (design @ beta)
    resid_std = float(np.nanstd(residuals, ddof=1)) if len(residuals) > 2 else float(np.nan)
    return pred, resid_std, int(len(y_train))


def _feature_frame(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    atr = df.get("atr", high - low).astype(float).replace(0, np.nan)
    price = close.replace(0, np.nan)

    bars_15m = _bars_for_minutes(settings, 15, 1)
    bars_30m = _bars_for_minutes(settings, 30, 1)
    bars_1h = _bars_for_minutes(settings, 60, 1)
    bars_4h = _bars_for_minutes(settings, 240, 1)
    bars_1d = _bars_for_minutes(settings, 1440, 1)

    range_window = max(bars_1d, 30)
    rolling_high = high.rolling(range_window, min_periods=min(30, range_window)).max()
    rolling_low = low.rolling(range_window, min_periods=min(30, range_window)).min()
    range_position = ((close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)).clip(0, 1)

    features = pd.DataFrame(index=df.index)
    features["ret_1"] = ret
    features["ret_15m"] = ret.rolling(bars_15m, min_periods=max(1, min(bars_15m, 3))).sum()
    features["ret_30m"] = ret.rolling(bars_30m, min_periods=max(1, min(bars_30m, 3))).sum()
    features["ret_1h"] = ret.rolling(bars_1h, min_periods=max(1, min(bars_1h, 6))).sum()
    features["ret_4h"] = ret.rolling(bars_4h, min_periods=max(1, min(bars_4h, 12))).sum()
    features["ret_1d"] = ret.rolling(bars_1d, min_periods=max(1, min(bars_1d, 24))).sum()
    features["vol_30m"] = ret.rolling(bars_30m, min_periods=max(2, min(bars_30m, 6))).std()
    features["vol_1h"] = ret.rolling(bars_1h, min_periods=max(2, min(bars_1h, 8))).std()
    features["vol_4h"] = ret.rolling(bars_4h, min_periods=max(2, min(bars_4h, 12))).std()
    features["vol_1d"] = ret.rolling(bars_1d, min_periods=max(2, min(bars_1d, 24))).std()
    features["body_atr"] = (close - open_) / atr
    features["range_atr"] = (high - low) / atr
    features["atr_pct"] = atr / price
    features["ema_gap"] = (df.get("ema_fast", close).astype(float) - df.get("ema_slow", close).astype(float)) / price
    features["trend_gap"] = (close - df.get("trend_sma", close).astype(float)) / price
    features["vwap_gap"] = (close - df.get("vwap", close).astype(float)) / price
    features["rsi_norm"] = (df.get("rsi", pd.Series(50, index=df.index)).astype(float) - 50.0) / 50.0
    features["adx_norm"] = df.get("adx", pd.Series(0, index=df.index)).astype(float) / 100.0
    features["range_position"] = range_position
    features["squeeze_on"] = df.get("squeeze_on", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(float)
    features["squeeze_recent"] = df.get("squeeze_recent", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(float)
    features["release_bullish"] = df.get("release_bullish_confirmed", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(float)
    features["release_bearish"] = df.get("release_bearish_confirmed", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(float)
    features["kc_momentum"] = df.get("kc_momentum", pd.Series(0, index=df.index)).astype(float)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fallback_sigma(returns: pd.Series, steps: int) -> float:
    window = min(max(steps * 8, 48), 576)
    base = returns.tail(window).std()
    if not math.isfinite(float(base)) or float(base) <= 0:
        base = returns.std()
    if not math.isfinite(float(base)) or float(base) <= 0:
        return 0.0001 * math.sqrt(max(steps, 1))
    return float(base) * math.sqrt(max(steps, 1))


def _min_ev_points(label: str, settings: dict) -> float:
    override_key = f"min_ev_points_{label}"
    if override_key in settings:
        return float(settings.get(override_key, DEFAULT_MIN_EV_POINTS.get(label, 0.8)))
    return float(settings.get("min_ev_points", DEFAULT_MIN_EV_POINTS.get(label, 0.8)))


def _bias_from_probs(p_up: float, p_down: float, settings: dict, mode: str) -> str:
    gap_key = "filter_min_probability_gap" if mode == "filter" else "horizon_min_probability_gap"
    prob_gap = float(settings.get(gap_key, 0.08 if mode == "filter" else 0.06))
    if p_up > p_down + prob_gap:
        return "bullish"
    if p_down > p_up + prob_gap:
        return "bearish"
    return "mixed"


def _validation_prediction(mu: float, sigma: float, cost_return: float, settings: dict, mode: str) -> str:
    sigma = max(float(sigma), 1e-6)
    p_up = 1.0 - _norm_cdf((cost_return - mu) / sigma)
    p_down = _norm_cdf((-cost_return - mu) / sigma)
    return _bias_from_probs(p_up, p_down, settings, mode)


def _actual_label(y_value: float, cost_return: float) -> str:
    if y_value > cost_return:
        return "bullish"
    if y_value < -cost_return:
        return "bearish"
    return "mixed"


def _walk_forward_validation(
    features: pd.DataFrame,
    target: pd.Series,
    close: pd.Series,
    settings: dict,
    mode: str,
    steps: int,
    cost_points: float,
) -> dict:
    max_tests = int(settings.get("walk_forward_tests", 80))
    min_train = int(settings.get("walk_forward_min_train", 300))
    lookback = int(settings.get("horizon_model_lookback_bars", 3000))
    alpha = float(settings.get("horizon_ridge_alpha", 2.0))

    last_target_idx = max(len(target) - steps, 0)
    if last_target_idx <= min_train:
        return _empty_validation(0)

    start_idx = max(min_train, last_target_idx - max_tests)
    tested = 0
    bull_total = bull_hits = 0
    bear_total = bear_hits = 0
    mixed_total = mixed_hits = 0
    active_total = active_hits = 0
    abs_errors_points: list[float] = []

    for idx in range(start_idx, last_target_idx):
        train_start = max(0, idx - lookback)
        x_train = features.iloc[train_start:idx].to_numpy(dtype=float)
        y_train = target.iloc[train_start:idx].to_numpy(dtype=float)
        x_now = features.iloc[idx].to_numpy(dtype=float)
        y_now = target.iloc[idx]
        price_now = _safe_float(close.iloc[idx], 0.0)
        if not math.isfinite(float(y_now)) or price_now <= 0:
            continue

        mu, resid_std, train_rows = _ridge_fit_predict(x_train, y_train, x_now, alpha=alpha)
        if train_rows < min_train:
            continue
        sigma = resid_std if math.isfinite(float(resid_std)) and float(resid_std) > 0 else _fallback_sigma(pd.Series(y_train).dropna(), steps)
        cost_return = float(cost_points) / max(price_now, 0.01)
        predicted = _validation_prediction(mu, sigma, cost_return, settings, mode)
        actual = _actual_label(float(y_now), cost_return)
        tested += 1
        abs_errors_points.append(abs(float(y_now) - float(mu)) * price_now)

        if predicted == "bullish":
            bull_total += 1
            active_total += 1
            if actual == "bullish":
                bull_hits += 1
                active_hits += 1
        elif predicted == "bearish":
            bear_total += 1
            active_total += 1
            if actual == "bearish":
                bear_hits += 1
                active_hits += 1
        else:
            mixed_total += 1
            if actual == "mixed":
                mixed_hits += 1

    return {
        "tested": tested,
        "bull_samples": bull_total,
        "bear_samples": bear_total,
        "mixed_samples": mixed_total,
        "bull_accuracy": _pct(bull_hits, bull_total),
        "bear_accuracy": _pct(bear_hits, bear_total),
        "mixed_accuracy": _pct(mixed_hits, mixed_total),
        "active_accuracy": _pct(active_hits, active_total),
        "active_samples": active_total,
        "mae_points": round(float(np.mean(abs_errors_points)), 2) if abs_errors_points else None,
    }


def _pct(hits: int, total: int):
    if total <= 0:
        return None
    return round((hits / total) * 100.0, 1)


def _empty_validation(tested: int) -> dict:
    return {
        "tested": tested,
        "bull_samples": 0,
        "bear_samples": 0,
        "mixed_samples": 0,
        "bull_accuracy": None,
        "bear_accuracy": None,
        "mixed_accuracy": None,
        "active_accuracy": None,
        "active_samples": 0,
        "mae_points": None,
    }


def _side_validated(validation: dict, side: str, settings: dict) -> bool:
    min_samples = int(settings.get("walk_forward_min_side_samples", 8))
    min_accuracy = float(settings.get("walk_forward_min_accuracy", 55.0))
    samples_key = "bull_samples" if side == "bullish" else "bear_samples"
    accuracy_key = "bull_accuracy" if side == "bullish" else "bear_accuracy"
    accuracy = validation.get(accuracy_key)
    return int(validation.get(samples_key, 0) or 0) >= min_samples and accuracy is not None and float(accuracy) >= min_accuracy


def _reliability(validation: dict, settings: dict) -> str:
    min_tests = int(settings.get("walk_forward_min_tests", 30))
    min_accuracy = float(settings.get("walk_forward_min_accuracy", 55.0))
    active_accuracy = validation.get("active_accuracy")
    if int(validation.get("tested", 0) or 0) < min_tests:
        return "Unproven"
    if active_accuracy is None:
        return "No active history"
    if float(active_accuracy) >= min_accuracy:
        return "Validated"
    return "Failed validation"


def _decision(label: str, role: str, mode: str, bias: str, ev_buy: float, ev_sell: float, validation: dict, settings: dict) -> str:
    if bool(settings.get("signals_disabled", False)):
        return "Disabled - sample data"

    if mode == "filter":
        if bias in {"bullish", "bearish"} and _side_validated(validation, bias, settings):
            return f"{role}: validated {bias} pressure"
        if bias in {"bullish", "bearish"}:
            return f"{role}: unproven {bias} pressure"
        return f"{role}: mixed / no block"

    min_ev = _min_ev_points(label, settings)
    if bias == "bullish" and ev_buy >= min_ev and _side_validated(validation, "bullish", settings):
        return "Validated buy edge"
    if bias == "bearish" and ev_sell >= min_ev and _side_validated(validation, "bearish", settings):
        return "Validated sell edge"
    if bias in {"bullish", "bearish"}:
        return "Unproven edge - wait"
    return "No edge"


def _forecast_one(df: pd.DataFrame, features: pd.DataFrame, settings: dict, label: str, config: dict) -> dict:
    horizon_minutes = int(config["minutes"])
    role = str(config["role"])
    mode = str(config["mode"])
    base_minutes = _base_minutes(settings)
    price = _safe_float(df["close"].iloc[-1], 0.0)
    min_ev = _min_ev_points(label, settings)

    if base_minutes > horizon_minutes or horizon_minutes % base_minutes != 0:
        return {
            "horizon": label,
            "role": role,
            "status": "unavailable",
            "bias": "N/A",
            "up_probability": None,
            "down_probability": None,
            "flat_probability": None,
            "bull_confidence": None,
            "bear_confidence": None,
            "validation_status": "Unproven",
            "validation_tests": 0,
            "expected_return_pct": None,
            "expected_low": None,
            "expected_high": None,
            "ev_buy_points": None,
            "ev_sell_points": None,
            "min_ev_points": min_ev if mode == "entry" else None,
            "decision": f"Use 5m base timeframe for this horizon",
            "expiry": label,
            "training_rows": 0,
            "method": "requires 5m/even base timeframe",
        }

    steps = max(int(horizon_minutes / base_minutes), 1)
    close = df["close"].astype(float)
    target = np.log(close.shift(-steps) / close).replace([np.inf, -np.inf], np.nan)

    train_cutoff = max(len(df) - steps, 0)
    lookback = int(settings.get("horizon_model_lookback_bars", 3000))
    start = max(0, train_cutoff - lookback)
    x_train = features.iloc[start:train_cutoff].to_numpy(dtype=float)
    y_train = target.iloc[start:train_cutoff].to_numpy(dtype=float)
    x_now = features.iloc[-1].to_numpy(dtype=float)

    mu, resid_std, training_rows = _ridge_fit_predict(x_train, y_train, x_now, alpha=float(settings.get("horizon_ridge_alpha", 2.0)))
    returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    sigma = resid_std if math.isfinite(float(resid_std)) and float(resid_std) > 0 else _fallback_sigma(returns, steps)
    sigma = max(float(sigma), 1e-6)

    z = float(settings.get("horizon_range_z", 1.28))
    lower = price * math.exp(mu - z * sigma)
    upper = price * math.exp(mu + z * sigma)

    cost_points = float(settings.get("horizon_cost_points", 0.30))
    cost_return = cost_points / max(price, 0.01)
    p_up = 1.0 - _norm_cdf((cost_return - mu) / sigma)
    p_down = _norm_cdf((-cost_return - mu) / sigma)
    p_flat = max(0.0, 1.0 - p_up - p_down)

    up_move = max(upper - price, 0.0)
    down_move = max(price - lower, 0.0)
    ev_buy = p_up * up_move - p_down * down_move - cost_points
    ev_sell = p_down * down_move - p_up * up_move - cost_points
    bias = _bias_from_probs(p_up, p_down, settings, mode)
    validation = _walk_forward_validation(features, target, close, settings, mode, steps, cost_points)
    validation_status = _reliability(validation, settings)

    return {
        "horizon": label,
        "role": role,
        "status": "ok",
        "bias": bias,
        "up_probability": validation.get("bull_accuracy"),
        "down_probability": validation.get("bear_accuracy"),
        "flat_probability": validation.get("mixed_accuracy"),
        "bull_confidence": validation.get("bull_accuracy"),
        "bear_confidence": validation.get("bear_accuracy"),
        "validation_status": validation_status,
        "validation_tests": validation.get("tested"),
        "bull_validation_samples": validation.get("bull_samples"),
        "bear_validation_samples": validation.get("bear_samples"),
        "active_accuracy": validation.get("active_accuracy"),
        "active_samples": validation.get("active_samples"),
        "mae_points": validation.get("mae_points"),
        "model_up_probability": round(p_up * 100.0, 1),
        "model_down_probability": round(p_down * 100.0, 1),
        "model_flat_probability": round(p_flat * 100.0, 1),
        "expected_return_pct": round((math.exp(mu) - 1.0) * 100.0, 4),
        "expected_low": round(lower, 2),
        "expected_high": round(upper, 2),
        "ev_buy_points": round(ev_buy, 2),
        "ev_sell_points": round(ev_sell, 2),
        "min_ev_points": round(min_ev, 2) if mode == "entry" else None,
        "decision": _decision(label, role, mode, bias, ev_buy, ev_sell, validation, settings),
        "expiry": label,
        "training_rows": training_rows,
        "method": "rolling ridge return + walk-forward validation; confidence is recent validated hit-rate",
    }


def build_multi_horizon_forecast(df: pd.DataFrame, settings: dict) -> list[dict]:
    """Build walk-forward validated 5m signal plus 1h/4h/Daily filters from enriched OHLC data."""
    if df is None or df.empty or len(df) < 120:
        return [{
            "horizon": label,
            "role": config["role"],
            "status": "insufficient_data",
            "bias": "N/A",
            "up_probability": None,
            "down_probability": None,
            "flat_probability": None,
            "bull_confidence": None,
            "bear_confidence": None,
            "validation_status": "Unproven",
            "validation_tests": 0,
            "expected_return_pct": None,
            "expected_low": None,
            "expected_high": None,
            "ev_buy_points": None,
            "ev_sell_points": None,
            "min_ev_points": _min_ev_points(label, settings) if config["mode"] == "entry" else None,
            "decision": "Need more clean bars",
            "expiry": label,
            "training_rows": 0,
            "method": "walk-forward validation unavailable",
        } for label, config in HORIZON_CONFIG.items()]

    features = _feature_frame(df, settings)
    return [_forecast_one(df, features, settings, label, config) for label, config in HORIZON_CONFIG.items()]
