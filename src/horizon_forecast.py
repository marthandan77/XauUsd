from __future__ import annotations

import math

import numpy as np
import pandas as pd


HORIZONS_MINUTES = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
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
    "15m": 0.80,
    "30m": 1.20,
    "1h": 1.80,
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


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_now: np.ndarray, alpha: float = 1.0) -> tuple[float, float, int]:
    """Fit a small ridge model with closed-form linear algebra and predict the latest row."""
    valid = np.isfinite(y_train) & np.isfinite(x_train).all(axis=1)
    x_train = x_train[valid]
    y_train = y_train[valid]

    if len(y_train) < max(80, x_train.shape[1] * 8):
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


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    atr = df.get("atr", high - low).astype(float).replace(0, np.nan)
    price = close.replace(0, np.nan)

    rolling_high = high.rolling(288, min_periods=30).max()
    rolling_low = low.rolling(288, min_periods=30).min()
    range_position = ((close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)).clip(0, 1)

    features = pd.DataFrame(index=df.index)
    features["ret_1"] = ret
    features["ret_3"] = ret.rolling(3, min_periods=2).sum()
    features["ret_6"] = ret.rolling(6, min_periods=3).sum()
    features["ret_12"] = ret.rolling(12, min_periods=6).sum()
    features["vol_6"] = ret.rolling(6, min_periods=3).std()
    features["vol_12"] = ret.rolling(12, min_periods=6).std()
    features["vol_24"] = ret.rolling(24, min_periods=12).std()
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
    base = returns.tail(144).std()
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


def _decision(label: str, p_up: float, p_down: float, ev_buy: float, ev_sell: float, settings: dict) -> str:
    if bool(settings.get("signals_disabled", False)):
        return "Disabled - sample data"
    min_ev = _min_ev_points(label, settings)
    prob_gap = float(settings.get("horizon_min_probability_gap", 0.06))
    if p_up >= p_down + prob_gap and ev_buy >= min_ev:
        return "Buy edge"
    if p_down >= p_up + prob_gap and ev_sell >= min_ev:
        return "Sell edge"
    if max(p_up, p_down) >= 0.55:
        return "Watch"
    return "No edge"


def _forecast_one(df: pd.DataFrame, features: pd.DataFrame, settings: dict, label: str, horizon_minutes: int) -> dict:
    interval = str(settings.get("price_interval", "5m"))
    base_minutes = INTERVAL_MINUTES.get(interval, 5)
    price = _safe_float(df["close"].iloc[-1], 0.0)
    min_ev = _min_ev_points(label, settings)

    if base_minutes > horizon_minutes or horizon_minutes % base_minutes != 0:
        return {
            "horizon": label,
            "status": "unavailable",
            "bias": "N/A",
            "up_probability": None,
            "down_probability": None,
            "flat_probability": None,
            "expected_return_pct": None,
            "expected_low": None,
            "expected_high": None,
            "ev_buy_points": None,
            "ev_sell_points": None,
            "min_ev_points": min_ev,
            "decision": f"Use {base_minutes}m base or switch to 5m timeframe",
            "expiry": label,
            "training_rows": 0,
            "method": "requires finer/even timeframe",
        }

    steps = max(int(horizon_minutes / base_minutes), 1)
    close = df["close"].astype(float)
    target = np.log(close.shift(-steps) / close).replace([np.inf, -np.inf], np.nan)

    train_cutoff = max(len(df) - steps, 0)
    lookback = int(settings.get("horizon_model_lookback_bars", 1500))
    start = max(0, train_cutoff - lookback)
    x_train = features.iloc[start:train_cutoff].to_numpy(dtype=float)
    y_train = target.iloc[start:train_cutoff].to_numpy(dtype=float)
    x_now = features.iloc[-1].to_numpy(dtype=float)

    mu, resid_std, training_rows = _ridge_fit_predict(x_train, y_train, x_now, alpha=float(settings.get("horizon_ridge_alpha", 1.0)))
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

    prob_gap = float(settings.get("horizon_min_probability_gap", 0.06))
    if p_up > p_down + prob_gap:
        bias = "bullish"
    elif p_down > p_up + prob_gap:
        bias = "bearish"
    else:
        bias = "mixed"

    return {
        "horizon": label,
        "status": "ok",
        "bias": bias,
        "up_probability": round(p_up * 100.0, 1),
        "down_probability": round(p_down * 100.0, 1),
        "flat_probability": round(p_flat * 100.0, 1),
        "expected_return_pct": round((math.exp(mu) - 1.0) * 100.0, 4),
        "expected_low": round(lower, 2),
        "expected_high": round(upper, 2),
        "ev_buy_points": round(ev_buy, 2),
        "ev_sell_points": round(ev_sell, 2),
        "min_ev_points": round(min_ev, 2),
        "decision": _decision(label, p_up, p_down, ev_buy, ev_sell, settings),
        "expiry": label,
        "training_rows": training_rows,
        "method": "rolling ridge return + residual volatility + EV threshold",
    }


def build_multi_horizon_forecast(df: pd.DataFrame, settings: dict) -> list[dict]:
    """Build formula-based 5m/15m/30m/1h forecasts from enriched OHLC data.

    The model is trained on the currently loaded historical bars during the scan. It is
    not a fixed rule score: targets are forward log-returns for each horizon.
    """
    if df is None or df.empty or len(df) < 120:
        return [{
            "horizon": label,
            "status": "insufficient_data",
            "bias": "N/A",
            "up_probability": None,
            "down_probability": None,
            "flat_probability": None,
            "expected_return_pct": None,
            "expected_low": None,
            "expected_high": None,
            "ev_buy_points": None,
            "ev_sell_points": None,
            "min_ev_points": _min_ev_points(label, settings),
            "decision": "Need more clean bars",
            "expiry": label,
            "training_rows": 0,
            "method": "rolling ridge return + residual volatility + EV threshold",
        } for label in HORIZONS_MINUTES]

    features = _feature_frame(df)
    return [_forecast_one(df, features, settings, label, minutes) for label, minutes in HORIZONS_MINUTES.items()]
