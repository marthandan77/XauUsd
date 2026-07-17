import numpy as np
import pandas as pd

from src.quant_forecast import build_feature_frame, build_unified_forecast


def _bars(rows=900):
    index = pd.date_range("2026-01-01", periods=rows, freq="15min")
    rng = np.random.default_rng(4)
    close = 2300 + np.cumsum(rng.normal(0.03, 1.0, rows))
    frame = pd.DataFrame(index=index)
    frame["close"] = close
    frame["open"] = close + rng.normal(0, 0.3, rows)
    frame["high"] = np.maximum(frame["open"], frame["close"]) + 0.8
    frame["low"] = np.minimum(frame["open"], frame["close"]) - 0.8
    frame["ema_fast"] = frame["close"].ewm(span=20).mean()
    frame["ema_slow"] = frame["close"].ewm(span=50).mean()
    frame["trend_sma"] = frame["close"].rolling(50, min_periods=5).mean()
    frame["atr"] = 2.0
    frame["vwap"] = frame["close"].expanding().mean()
    frame["rsi"] = 50 + frame["close"].diff().rolling(14, min_periods=3).mean().fillna(0)
    frame["adx"] = 25.0
    frame["kc_momentum"] = frame["close"] - frame["close"].rolling(20, min_periods=5).mean()
    frame["kc_momentum_prev"] = frame["kc_momentum"].shift(1)
    frame["realized_vol"] = np.log(frame["close"] / frame["close"].shift(1)).rolling(30, min_periods=10).std()
    frame["squeeze_on"] = False
    frame["squeeze_fired"] = False
    return frame.dropna()


def test_feature_frame_contains_quant_inputs():
    bars = _bars()
    features = build_feature_frame(bars, {"price_interval": "15m", "lookback_days": 7})
    assert "ema_spread" in features
    assert "range_position" in features
    assert "kc_momentum_atr" in features


def test_unified_forecast_returns_price_distribution():
    bars = _bars()
    settings = {
        "price_interval": "15m",
        "lookback_days": 7,
        "quant_min_samples": 100,
        "quant_default_horizon_bars": 4,
        "quant_candidate_horizons": [1, 2, 4],
        "adx_minimum": 20,
        "rsi_level": 50,
    }
    scores = {"bull_score": 84, "bear_score": 30}
    plan = {"tp1": float(bars.close.iloc[-1] + 6), "stop": float(bars.close.iloc[-1] - 4)}
    veto = {"final_action": "BUY PLAN"}
    result = build_unified_forecast(bars, scores, plan, veto, settings, pd.Timestamp("2026-02-01", tz="UTC"))
    assert result["ready"]
    assert result["lower_price"] < result["upper_price"]
    assert 0 <= result["probability_up"] <= 1
    assert result["horizon_bars"] in {1, 2, 4}
