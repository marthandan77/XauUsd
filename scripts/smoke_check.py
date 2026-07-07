from __future__ import annotations

from src.bias_validation import bias_validation_summary
from src.data_feed import make_sample_bars
from src.exhaustion_engine import exhaustion_context
from src.forecast_engine import choose_action, score_forecast
from src.horizon_forecaster import multi_horizon_forecast
from src.indicators import add_indicators
from src.kc_squeeze_engine import kc_squeeze_summary
from src.macro_engine import macro_context
from src.market_map import build_market_map
from src.multi_timeframe import multi_timeframe_confirmation
from src.regime_engine import classify_regime
from src.scalp_gate import scalp_gate
from src.trade_plan import build_trade_plan
from src.veto_engine import apply_veto


def main() -> None:
    settings = {
        "price_interval": "5m",
        "lookback_days": 7,
        "ema_fast": 50,
        "ema_slow": 200,
        "trend_length": 200,
        "atr_period": 14,
        "adx_minimum": 20,
        "buy_threshold": 78,
        "sell_threshold": 78,
        "wait_threshold": 60,
        "kc_squeeze_enabled": True,
        "bb_length": 20,
        "bb_mult": 2.0,
        "kc_length": 20,
        "kc_mult": 1.5,
        "kc_breakout_score": 25,
        "kc_momentum_score": 10,
        "kc_compression_score": 3,
        "realized_vol_length": 100,
        "middle_range_lower_pct": 35,
        "middle_range_upper_pct": 65,
        "scalp_min_samples": 40,
        "scalp_min_validation_samples": 15,
        "scalp_validation_fraction": 0.30,
        "scalp_cost_buffer": 0.40,
        "scalp_room_ratio_min": 1.50,
        "scalp_15m_block_edge": -0.25,
        "scalp_rsi_sell_floor": 35.0,
        "scalp_rsi_buy_ceiling": 65.0,
        "horizon_min_samples": 30,
        "horizon_tp1_quantile": 0.50,
        "horizon_tp2_quantile": 0.75,
        "horizon_adverse_quantile": 0.80,
        "long_plans_enabled": True,
        "short_plans_enabled": True,
        "min_reward_risk": 1.5,
    }
    bars = make_sample_bars(rows=5000, freq="5min")
    enriched = add_indicators(bars, settings)
    clean = enriched.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
    assert not clean.empty, "indicator warm-up produced no clean bars"

    market = build_market_map(clean, settings)
    regime = classify_regime(clean, market, settings)
    macro = macro_context("mixed", False)
    kc = kc_squeeze_summary(clean, settings)
    latest = clean.iloc[-1].to_dict()
    latest["kc_state"] = kc.get("state")
    latest["kc_reason"] = kc.get("reason")
    scores = score_forecast(latest, market, regime, macro, settings)
    action = choose_action(scores, settings)
    plan = build_trade_plan(action, market, settings)
    exhaustion = exhaustion_context(clean, market, settings)
    veto = apply_veto(action, plan, market, regime, macro, settings, exhaustion.to_dict())
    scalp = scalp_gate(clean, market, scores, macro, kc, settings)
    mtf = multi_timeframe_confirmation(clean, settings, macro)
    bias = bias_validation_summary(clean, market, scores, settings)
    multi_horizon_forecast(clean, market, action, veto, "NO CANDIDATE", settings)

    assert scores["bias"] in {"bullish", "bearish", "mixed"}
    assert veto["final_action"] in {"BUY PLAN", "SELL PLAN", "WAIT", "HOLD", "EXIT LONG / AVOID BUY"}
    assert scalp.recommendation in {"BUY SCALP", "SELL SCALP", "NO SCALP"}
    assert mtf["verdict"] in {"ALIGNED", "SCALP_ONLY", "CONFLICT", "INCOMPLETE"}
    assert bias["overall_verdict"] in {"STRONG", "WEAK", "NO EDGE"}
    print("Smoke check passed")


if __name__ == "__main__":
    main()
