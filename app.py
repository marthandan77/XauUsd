from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.ai_explainer import explain
from src.bias_validation import bias_validation_summary, bias_validation_table
from src.data_feed import FeedResult, load_csv, load_live_bars, make_sample_bars
from src.exhaustion_engine import exhaustion_context
from src.forecast_engine import choose_action, score_forecast
from src.horizon_forecaster import horizon_forecast_table, multi_horizon_forecast
from src.indicators import add_indicators
from src.kc_squeeze_engine import kc_squeeze_summary
from src.macro_engine import macro_context
from src.market_map import build_market_map
from src.multi_timeframe import multi_timeframe_confirmation, multi_timeframe_table
from src.range_model import price_bands
from src.regime_engine import classify_regime
from src.scalp_gate import scalp_gate, scalp_horizon_table
from src.scan_logger import append_scan_log, read_scan_log
from src.trade_plan import advisory_position_size, build_trade_plan
from src.veto_engine import apply_veto

ROOT = Path(__file__).resolve().parent
RUNTIME_SETTINGS_PATH = ROOT / "data" / "runtime_settings.yaml"
ACTIONABLE = {"BUY PLAN", "SELL PLAN"}
SCALP_ACTIONS = {"BUY SCALP", "SELL SCALP"}
PERSISTED_SETTING_KEYS = {
    "price_interval", "price_period", "data_mode", "macro_bias",
    "buy_threshold", "sell_threshold", "wait_threshold", "atr_multiplier", "min_reward_risk",
    "middle_range_lower_pct", "middle_range_upper_pct", "kc_squeeze_enabled",
    "bb_length", "bb_mult", "kc_length", "kc_mult", "trend_length", "atr_period",
    "atr_stop_multiplier", "atr_tp_multiplier", "sell_tp1_atr_multiplier", "sell_tp2_atr_multiplier",
    "sell_support_buffer_atr", "horizon_min_samples", "horizon_tp1_quantile", "horizon_tp2_quantile",
    "horizon_adverse_quantile", "scalp_min_samples", "scalp_cost_buffer", "scalp_rsi_sell_floor",
    "scalp_rsi_buy_ceiling", "risk_per_trade_pct", "long_plans_enabled", "short_plans_enabled",
    "show_bollinger_bands", "show_keltner_channels", "news_block_manual",
}

st.set_page_config(page_title="XAU/USD Forecast Manager", layout="wide")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def load_settings() -> dict:
    settings = load_yaml(ROOT / "config/default_settings.yaml")
    saved = load_yaml(RUNTIME_SETTINGS_PATH)
    if saved:
        settings.update({k: v for k, v in saved.items() if k in PERSISTED_SETTING_KEYS})
    return settings


def save_runtime_settings(settings: dict) -> None:
    RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = {k: settings[k] for k in sorted(PERSISTED_SETTING_KEYS) if k in settings}
    RUNTIME_SETTINGS_PATH.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")


def reset_runtime_settings() -> None:
    if RUNTIME_SETTINGS_PATH.exists():
        RUNTIME_SETTINGS_PATH.unlink()


def _select_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _sample_freq(settings: dict) -> str:
    return {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1d"}.get(str(settings.get("price_interval", "15m")), "15min")


def fmt_price(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    return f"{value:,.2f}" if math.isfinite(value) else "N/A"


def fmt_units(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    return f"{value:,.2f}" if math.isfinite(value) and value > 0 else "N/A"


def fmt_metric(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    return f"{value:.{digits}f}" if math.isfinite(value) else "N/A"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if math.isfinite(number) else default


def has_plan_levels(plan: dict) -> bool:
    return plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None


def plan_value(plan: dict, key: str):
    return plan.get(key) if has_plan_levels(plan) else None


def candidate_plan_from_scores(scores: dict, market, settings: dict) -> tuple[str, dict, str]:
    threshold = int(settings.get("forecast_threshold", 65))
    bias = str(scores.get("bias", "mixed"))
    bull_score = int(scores.get("bull_score", 0))
    bear_score = int(scores.get("bear_score", 0))
    if bias == "bullish" and bull_score >= threshold and bool(settings.get("long_plans_enabled", True)):
        plan = build_trade_plan("BUY PLAN", market, settings)
        plan["note"] = "Bullish forecast candidate. Use only if final action is BUY PLAN and veto filters are clean."
        return "BUY CANDIDATE", plan, "Bullish candidate levels shown because bull forecast is above forecast threshold."
    if bias == "bearish" and bear_score >= threshold:
        preview_settings = dict(settings)
        preview_settings["short_plans_enabled"] = True
        plan = build_trade_plan("SELL PLAN", market, preview_settings)
        plan["note"] = "Bearish forecast candidate. Use only if final action is SELL PLAN and veto filters are clean."
        return "SELL CANDIDATE", plan, "Bearish candidate levels shown because bear forecast is above forecast threshold."
    return "NO CANDIDATE", {"entry_zone_low": None, "entry_zone_high": None, "stop": None, "tp1": None, "tp2": None, "risk": 0.0, "note": "No forecast candidate above threshold."}, ""


def plan_status(action: str, final_action: str, plan: dict, candidate_action: str) -> str:
    if final_action in ACTIONABLE:
        return f"Active {final_action}"
    if action in ACTIONABLE and has_plan_levels(plan):
        return f"Rejected {action}"
    if candidate_action != "NO CANDIDATE":
        return candidate_action
    return "No entry plan"


def refresh_panel() -> None:
    st.sidebar.header("Refresh")
    if st.sidebar.button("Manual refresh now", type="primary"):
        st.rerun()
    st.sidebar.caption("Auto-refresh is disabled. The app refreshes only when you press the button or manually reload the browser.")


def persistence_panel(settings: dict) -> None:
    st.sidebar.header("Settings memory")
    st.sidebar.caption("Temporary changes affect the current screen only. They persist after browser refresh only when Save is clicked.")
    if st.sidebar.button("Save sidebar settings", type="primary"):
        save_runtime_settings(settings)
        st.sidebar.success("Settings saved.")
    if st.sidebar.button("Reset saved sidebar settings"):
        reset_runtime_settings()
        st.rerun()


def settings_panel(settings: dict, presets: dict) -> dict:
    st.sidebar.header("Forecast controls")
    if presets:
        selected = st.sidebar.selectbox("Preset", list(presets.keys()), index=0)
        if st.sidebar.button("Load preset"):
            settings.update(presets[selected])
    interval_options = ["5m", "15m", "30m", "1h", "4h", "1d"]
    period_options = ["7d", "30d", "60d", "6mo", "1y", "2y"]
    settings["price_interval"] = st.sidebar.selectbox("Timeframe", interval_options, index=_select_index(interval_options, str(settings.get("price_interval", "15m"))))
    settings["price_period"] = st.sidebar.selectbox("Lookback period", period_options, index=_select_index(period_options, str(settings.get("price_period", "30d"))))
    settings["buy_threshold"] = st.sidebar.slider("Bull threshold", 50, 95, int(settings.get("buy_threshold", 78)))
    settings["sell_threshold"] = st.sidebar.slider("Bear threshold", 50, 95, int(settings.get("sell_threshold", 78)))
    settings["wait_threshold"] = st.sidebar.slider("Watch threshold", 40, 90, int(settings.get("wait_threshold", 60)))
    settings["atr_multiplier"] = st.sidebar.slider("ATR guard multiplier", 0.8, 2.5, float(settings.get("atr_multiplier", 1.3)), 0.1)
    settings["min_reward_risk"] = st.sidebar.slider("Minimum room ratio", 1.0, 3.0, float(settings.get("min_reward_risk", 1.5)), 0.1)
    settings["middle_range_lower_pct"] = st.sidebar.slider("Middle zone lower %", 10, 49, int(settings.get("middle_range_lower_pct", 35)))
    settings["middle_range_upper_pct"] = st.sidebar.slider("Middle zone upper %", 51, 90, int(settings.get("middle_range_upper_pct", 65)))
    st.sidebar.header("KC Squeeze")
    settings["kc_squeeze_enabled"] = st.sidebar.toggle("Enable KC Squeeze module", bool(settings.get("kc_squeeze_enabled", True)))
    settings["bb_length"] = st.sidebar.slider("BB length", 10, 60, int(settings.get("bb_length", 20)))
    settings["bb_mult"] = st.sidebar.slider("BB multiplier", 1.0, 3.5, float(settings.get("bb_mult", 2.0)), 0.1)
    settings["kc_length"] = st.sidebar.slider("KC length", 10, 60, int(settings.get("kc_length", 20)))
    settings["kc_mult"] = st.sidebar.slider("KC multiplier", 1.0, 3.5, float(settings.get("kc_mult", 1.5)), 0.1)
    settings["trend_length"] = st.sidebar.slider("Trend SMA length", 50, 300, int(settings.get("trend_length", 200)))
    settings["atr_period"] = st.sidebar.slider("ATR period", 5, 50, int(settings.get("atr_period", 14)))
    settings["atr_stop_multiplier"] = st.sidebar.slider("ATR stop multiplier", 1.0, 5.0, float(settings.get("atr_stop_multiplier", 3.0)), 0.1)
    settings["atr_tp_multiplier"] = st.sidebar.slider("ATR take-profit multiplier", 1.0, 10.0, float(settings.get("atr_tp_multiplier", 6.0)), 0.1)
    settings["sell_tp1_atr_multiplier"] = st.sidebar.slider("Sell Take Profit 1 ATR", 0.50, 3.00, float(settings.get("sell_tp1_atr_multiplier", 1.75)), 0.05)
    settings["sell_tp2_atr_multiplier"] = st.sidebar.slider("Sell Take Profit 2 ATR", 0.75, 5.00, float(settings.get("sell_tp2_atr_multiplier", 3.25)), 0.05)
    settings["sell_support_buffer_atr"] = st.sidebar.slider("Sell support buffer ATR", 0.00, 1.00, float(settings.get("sell_support_buffer_atr", 0.10)), 0.05)
    settings["risk_per_trade_pct"] = st.sidebar.slider("Advisory risk %", 0.1, 2.0, float(settings.get("risk_per_trade_pct", 0.5)), 0.1)
    st.sidebar.header("Empirical Horizon Forecast")
    settings["horizon_min_samples"] = st.sidebar.slider("Minimum matching samples", 10, 200, int(settings.get("horizon_min_samples", 30)), 5)
    settings["horizon_tp1_quantile"] = st.sidebar.slider("TP1 historical quantile", 0.25, 0.75, float(settings.get("horizon_tp1_quantile", 0.50)), 0.05)
    settings["horizon_tp2_quantile"] = st.sidebar.slider("TP2 historical quantile", 0.50, 0.95, float(settings.get("horizon_tp2_quantile", 0.75)), 0.05)
    settings["horizon_adverse_quantile"] = st.sidebar.slider("Stop Loss adverse quantile", 0.50, 0.95, float(settings.get("horizon_adverse_quantile", 0.80)), 0.05)
    st.sidebar.header("Scalp Gate")
    settings["scalp_min_samples"] = st.sidebar.slider("Scalp minimum samples", 30, 200, int(settings.get("scalp_min_samples", 40)), 5)
    settings["scalp_cost_buffer"] = st.sidebar.slider("Scalp cost/slippage buffer", 0.00, 3.00, float(settings.get("scalp_cost_buffer", 0.40)), 0.05)
    settings["scalp_rsi_sell_floor"] = st.sidebar.slider("RSI sell floor", 20.0, 50.0, float(settings.get("scalp_rsi_sell_floor", 35.0)), 1.0)
    settings["scalp_rsi_buy_ceiling"] = st.sidebar.slider("RSI buy ceiling", 50.0, 80.0, float(settings.get("scalp_rsi_buy_ceiling", 65.0)), 1.0)
    settings["long_plans_enabled"] = True
    settings["short_plans_enabled"] = True
    st.sidebar.caption("BUY and SELL plan processing is always enabled.")
    settings["show_bollinger_bands"] = st.sidebar.toggle("Show Bollinger Bands", bool(settings.get("show_bollinger_bands", True)))
    settings["show_keltner_channels"] = st.sidebar.toggle("Show Keltner Channels", bool(settings.get("show_keltner_channels", True)))
    settings["news_block_manual"] = st.sidebar.toggle("Manual event block", bool(settings.get("news_block_manual", False)))
    return settings


def load_bars(settings: dict) -> FeedResult:
    st.sidebar.header("Data")
    data_mode_options = ["Twelve Data API", "CSV upload", "Sample"]
    data_mode = st.sidebar.radio("Data mode", data_mode_options, index=_select_index(data_mode_options, str(settings.get("data_mode", "Twelve Data API"))))
    settings["data_mode"] = data_mode
    if data_mode == "CSV upload":
        uploaded = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
        if uploaded is not None:
            return load_csv(uploaded)
        return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "CSV not uploaded; sample data used")
    if data_mode == "Twelve Data API":
        key_input = st.sidebar.text_input("Twelve Data API key - optional local session override", value="", type="password", help="Use this only if Streamlit cannot see your Windows environment variable. The key is not saved to GitHub or runtime settings.")
        if key_input.strip():
            settings["twelve_data_api_key_runtime"] = key_input.strip()
        feed = load_live_bars(settings)
        if not feed.bars.empty:
            return feed
        return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "Twelve Data unavailable; sample data used. " + feed.warning)
    return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "sample data")


def price_chart(df: pd.DataFrame, settings: dict, action: str, plan: dict) -> go.Figure:
    chart_df = df.tail(220)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df["open"], high=chart_df["high"], low=chart_df["low"], close=chart_df["close"], name="XAU/USD"))
    for column, label in [("ema_fast", "EMA Fast"), ("ema_slow", "EMA Slow"), ("trend_sma", "Trend SMA")]:
        if column in chart_df:
            fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[column], mode="lines", name=label))
    if bool(settings.get("show_bollinger_bands", True)):
        for column, label in [("bb_upper", "BB Upper"), ("bb_lower", "BB Lower")]:
            if column in chart_df:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[column], mode="lines", name=label))
    if bool(settings.get("show_keltner_channels", True)):
        for column, label in [("kc_upper", "KC Upper"), ("kc_lower", "KC Lower")]:
            if column in chart_df:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[column], mode="lines", name=label))
    if action in ACTIONABLE and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None:
        marker_price = (float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0
        fig.add_trace(go.Scatter(x=[chart_df.index[-1]], y=[marker_price], mode="markers+text", text=[action], textposition="top center", name="Advisory signal"))
    fig.update_layout(height=640, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def _display_horizon_table(forecasts) -> pd.DataFrame:
    table = horizon_forecast_table(forecasts)
    if table.empty:
        return table
    display = table.copy()
    for col in ["entry", "stop_loss", "take_profit_1", "take_profit_2", "tp1_move", "tp2_move", "adverse_move"]:
        if col in display:
            display[col] = display[col].apply(lambda value: None if pd.isna(value) else round(float(value), 2))
    if "historical_direction_probability_pct" in display:
        display["historical_direction_probability_pct"] = display["historical_direction_probability_pct"].apply(lambda value: None if pd.isna(value) else round(float(value), 1))
    return display


def _display_scalp_table(scalp_decision) -> pd.DataFrame:
    table = scalp_horizon_table(scalp_decision)
    if table.empty:
        return table
    display = table.copy()
    for col in ["entry", "stop_loss", "take_profit_1", "take_profit_2", "tp1_move", "tp2_move", "sl_move", "scalp_edge", "tp1_hit_rate_pct", "tp2_hit_rate_pct", "sl_hit_rate_pct"]:
        if col in display:
            display[col] = display[col].apply(lambda value: None if pd.isna(value) else round(float(value), 2))
    def explain_row(row: pd.Series) -> str:
        status = str(row.get("status", ""))
        if status != "VALID":
            return {"INSUFFICIENT_DATA": "Too few matches.", "INSUFFICIENT_VALIDATION": "Too few recent tests.", "NO_EDGE": "No positive edge.", "UNAVAILABLE": "Use 5m timeframe."}.get(status, "Check row.")
        edge = _safe_float(row.get("scalp_edge"), 0.0)
        tp1 = _safe_float(row.get("tp1_hit_rate_pct"), 0.0)
        sl = _safe_float(row.get("sl_hit_rate_pct"), 0.0)
        return "Good edge." if edge > 0 and tp1 > sl else "Negative edge." if edge < 0 else "Weak edge."
    display["explanation"] = display.apply(explain_row, axis=1)
    return display


def _horizon_value(scalp_display: pd.DataFrame, horizon: str, column: str):
    if scalp_display.empty or column not in scalp_display.columns:
        return None
    row = scalp_display[scalp_display["horizon"] == horizon]
    if row.empty:
        return None
    value = row.iloc[0].get(column)
    return None if pd.isna(value) else value


def _scalp_signal_confidence(scalp_decision, scalp_display: pd.DataFrame) -> int:
    if scalp_decision.recommendation not in SCALP_ACTIONS:
        return 0
    five_edge = _safe_float(_horizon_value(scalp_display, "5m", "scalp_edge"), 0.0)
    fifteen_edge = _safe_float(_horizon_value(scalp_display, "15m", "scalp_edge"), 0.0)
    tp1_rate = _safe_float(_horizon_value(scalp_display, "5m", "tp1_hit_rate_pct"), 0.0)
    sl_rate = _safe_float(_horizon_value(scalp_display, "5m", "sl_hit_rate_pct"), 0.0)
    train_samples = _safe_float(_horizon_value(scalp_display, "5m", "train_samples"), 0.0)
    validation_samples = _safe_float(_horizon_value(scalp_display, "5m", "validation_samples"), 0.0)
    room_ratio = _safe_float(scalp_decision.room_ratio, 0.0)
    confidence = 35.0 + min(max(five_edge, 0.0) * 12.0, 20.0)
    confidence += 15.0 if fifteen_edge >= 0 else 7.0
    confidence += min(max(tp1_rate - sl_rate, 0.0) * 0.35, 20.0)
    confidence += min(max(room_ratio - 1.5, 0.0) * 8.0, 10.0)
    confidence += min(train_samples / 40.0, 1.0) * 5.0 + min(validation_samples / 15.0, 1.0) * 5.0 + 5.0
    return int(max(0, min(round(confidence), 95)))


def _scalp_quality_label(confidence: int) -> str:
    if confidence >= 80:
        return "Strong"
    if confidence >= 70:
        return "Good"
    if confidence >= 60:
        return "Acceptable"
    return "Weak"


def _show_scalp_scan_result(scalp_decision, scalp_display: pd.DataFrame) -> str:
    confidence = _scalp_signal_confidence(scalp_decision, scalp_display)
    five_status = _horizon_value(scalp_display, "5m", "status")
    viable = scalp_decision.recommendation in SCALP_ACTIONS and str(five_status) == "VALID"
    five_edge = _safe_float(_horizon_value(scalp_display, "5m", "scalp_edge"), 0.0)
    fifteen_edge = _safe_float(_horizon_value(scalp_display, "15m", "scalp_edge"), 0.0)
    tp1_rate = _safe_float(_horizon_value(scalp_display, "5m", "tp1_hit_rate_pct"), 0.0)
    sl_rate = _safe_float(_horizon_value(scalp_display, "5m", "sl_hit_rate_pct"), 0.0)
    reasons = list(getattr(scalp_decision, "reasons", []) or [])
    if not viable:
        near_miss = str(five_status) == "VALID" and (five_edge > 0 or tp1_rate > sl_rate or _safe_float(scalp_decision.room_ratio, 0.0) > 0)
        if near_miss:
            st.warning("NEAR MISS / WAIT")
            st.write("Main blocker: " + (reasons[0] if reasons else "Scalp Gate has not cleared all filters."))
            cols = st.columns(6)
            cols[0].metric("Closest side", scalp_decision.side)
            cols[1].metric("5m Edge", fmt_metric(five_edge, 2))
            cols[2].metric("15m Edge", fmt_metric(fifteen_edge, 2))
            cols[3].metric("Room Ratio", fmt_metric(scalp_decision.room_ratio, 2))
            cols[4].metric("RSI", fmt_metric(scalp_decision.rsi, 1))
            cols[5].metric("KC Momentum", fmt_metric(scalp_decision.kc_momentum, 2))
            st.caption("No entry, SL or TP shown because at least one filter is still blocking the scalp.")
            return "NEAR MISS / WAIT"
        st.error("NO GOOD SIGNAL")
        if not reasons:
            reasons.append("Scalp Gate did not produce a clean signal.")
        st.write("Reason: " + "; ".join(reasons))
        return "NO GOOD SIGNAL"
    quality = _scalp_quality_label(confidence)
    st.success(f"SCALP SIGNAL: {scalp_decision.recommendation}")
    top = st.columns(5)
    top[0].metric("Confidence", f"{confidence}%", quality)
    top[1].metric("Entry", fmt_price(_horizon_value(scalp_display, "5m", "entry")))
    top[2].metric("Stop Loss", fmt_price(_horizon_value(scalp_display, "5m", "stop_loss")))
    top[3].metric("Take Profit 1", fmt_price(_horizon_value(scalp_display, "5m", "take_profit_1")))
    top[4].metric("Take Profit 2", fmt_price(_horizon_value(scalp_display, "5m", "take_profit_2")))
    quality_cols = st.columns(7)
    quality_cols[0].metric("5m Edge", fmt_metric(five_edge, 2))
    quality_cols[1].metric("15m Edge", fmt_metric(fifteen_edge, 2))
    quality_cols[2].metric("TP1 Hit %", fmt_metric(tp1_rate, 1))
    quality_cols[3].metric("TP2 Hit %", fmt_metric(_horizon_value(scalp_display, "5m", "tp2_hit_rate_pct"), 1))
    quality_cols[4].metric("SL Hit %", fmt_metric(sl_rate, 1))
    quality_cols[5].metric("Room Ratio", fmt_metric(scalp_decision.room_ratio, 2))
    quality_cols[6].metric("KC Momentum", fmt_metric(scalp_decision.kc_momentum, 2))
    st.caption("Reason: 5m trigger valid. 15m danger is clear or only slight caution. Room, RSI and KC guards passed.")
    return "SCALP SIGNAL"


def _scalp_value_guide(scalp_decision, scalp_display: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Recommendation", scalp_decision.recommendation, "Final answer."),
        ("Side", scalp_decision.side, "Allowed direction."),
        ("Room ratio", fmt_metric(scalp_decision.room_ratio, 2), "Scalp room / scalp risk."),
        ("RSI", fmt_metric(scalp_decision.rsi, 1), "Overbought/oversold guard."),
        ("KC momentum", fmt_metric(scalp_decision.kc_momentum, 2), "Negative=down, positive=up."),
        ("5m scalp edge", fmt_metric(_horizon_value(scalp_display, "5m", "scalp_edge"), 2), "Trigger. >0 good."),
        ("15m scalp edge", fmt_metric(_horizon_value(scalp_display, "15m", "scalp_edge"), 2), "Danger check. <=-0.25 blocks."),
        ("TP1 hit %", fmt_metric(_horizon_value(scalp_display, "5m", "tp1_hit_rate_pct"), 1), "Fast target odds."),
        ("TP2 hit %", fmt_metric(_horizon_value(scalp_display, "5m", "tp2_hit_rate_pct"), 1), "Stretch target odds."),
        ("SL hit %", fmt_metric(_horizon_value(scalp_display, "5m", "sl_hit_rate_pct"), 1), "Stop-loss odds."),
        ("Entry", fmt_price(_horizon_value(scalp_display, "5m", "entry")), "Current price used."),
        ("Stop Loss", fmt_price(_horizon_value(scalp_display, "5m", "stop_loss")), "Invalidation price."),
        ("Take Profit 1", fmt_price(_horizon_value(scalp_display, "5m", "take_profit_1")), "Fast target."),
        ("Take Profit 2", fmt_price(_horizon_value(scalp_display, "5m", "take_profit_2")), "Stretch target."),
    ]
    return pd.DataFrame(rows, columns=["Value", "Now", "Very short meaning"])


def _score_breakdown_table(scores: dict) -> pd.DataFrame:
    rows = []
    for bucket, values in dict(scores.get("components", {})).items():
        rows.append({"component": bucket, "bull": values.get("bull", 0), "bear": values.get("bear", 0), "net": values.get("bull", 0) - values.get("bear", 0)})
    return pd.DataFrame(rows)


def _scan_log_row(outcome: str, scalp_decision, scalp_display: pd.DataFrame, market, scores: dict, veto: dict, mtf: dict, exhaustion: dict) -> dict:
    return {
        "time_utc": pd.Timestamp.utcnow().isoformat(),
        "price": market.price,
        "outcome": outcome,
        "scalp_recommendation": scalp_decision.recommendation,
        "side": scalp_decision.side,
        "entry": _horizon_value(scalp_display, "5m", "entry"),
        "stop_loss": _horizon_value(scalp_display, "5m", "stop_loss"),
        "take_profit_1": _horizon_value(scalp_display, "5m", "take_profit_1"),
        "take_profit_2": _horizon_value(scalp_display, "5m", "take_profit_2"),
        "5m_edge": _horizon_value(scalp_display, "5m", "scalp_edge"),
        "15m_edge": _horizon_value(scalp_display, "15m", "scalp_edge"),
        "room_ratio": scalp_decision.room_ratio,
        "bias": scores.get("bias"),
        "bull_score": scores.get("bull_score"),
        "bear_score": scores.get("bear_score"),
        "final_action": veto.get("final_action"),
        "mtf_verdict": mtf.get("verdict"),
        "exhaustion_state": exhaustion.get("state"),
        "reasons": "; ".join(list(getattr(scalp_decision, "reasons", []) or [])),
    }


settings = load_settings()
presets = load_yaml(ROOT / "config/presets.yaml")
settings = settings_panel(settings, presets)
refresh_panel()
macro_options = ["mixed", "supportive", "restrictive"]
macro_bias = st.sidebar.selectbox("Macro bias", macro_options, index=_select_index(macro_options, str(settings.get("macro_bias", "mixed"))))
settings["macro_bias"] = macro_bias
feed = load_bars(settings)
persistence_panel(settings)

bars_raw = add_indicators(feed.bars, settings)
bars = bars_raw.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
if bars.empty:
    st.error("Not enough clean OHLC data after indicator warm-up. Increase lookback period or use sample data.")
    st.stop()

kc = kc_squeeze_summary(bars, settings)
latest_row = bars.iloc[-1].to_dict()
latest_row["kc_state"] = kc["state"]
latest_row["kc_reason"] = kc["reason"]
market = build_market_map(bars, settings)
regime = classify_regime(bars, market, settings)
macro = macro_context(macro_bias, bool(settings.get("news_block_manual", False)))
exhaustion = exhaustion_context(bars, market, settings)
exhaustion_dict = exhaustion.to_dict()
scores = score_forecast(latest_row, market, regime, macro, settings)
if kc["state"] != "insufficient_data":
    scores["kc_state"] = kc["state"]
    scores["kc_reason"] = kc["reason"]
action = choose_action(scores, settings)
plan = build_trade_plan(action, market, settings)
candidate_action, candidate_plan, candidate_note = candidate_plan_from_scores(scores, market, settings)
bands = price_bands(market, regime)
veto = apply_veto(action, plan, market, regime, macro, settings, exhaustion_dict)
summary = explain(regime, scores, veto, market, macro)
active_plan = veto["final_action"] in ACTIONABLE
status = plan_status(action, veto["final_action"], plan, candidate_action)
display_plan = plan if has_plan_levels(plan) else candidate_plan
horizon_forecasts = multi_horizon_forecast(bars, market, action, veto, candidate_action, settings)
scalp_decision = scalp_gate(bars, market, scores, macro, kc, settings)
bias_validation = bias_validation_summary(bars, market, scores, settings)
mtf_confirmation = multi_timeframe_confirmation(bars, settings, macro)

advisory_qty = 0.0
if active_plan and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None:
    advisory_qty = advisory_position_size(account_equity=10000, risk_pct=float(settings.get("risk_per_trade_pct", 0.5)) / 100.0, entry=(float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0, stop=float(plan["stop"]))

st.title("XAU/USD Forecast Manager")
st.caption("Advisory dashboard only. No broker execution. No auto-trading. No backtesting. Veto first, signal second.")
page = st.sidebar.radio("Page", ["Sniper Dashboard", "Forecast Manager", "Bull/Bear Validation", "MTF Confirmation", "Structure Levels", "Score Breakdown", "Exhaustion Guard", "Scalp Gate", "Multi-Horizon Forecast", "KC Squeeze", "Market Map", "Macro / News", "Settings", "Scan History", "Log Snapshot"])

source_cols = st.columns(4)
source_cols[0].metric("Data source", feed.source)
source_cols[1].metric("Latest price", fmt_price(market.price))
source_cols[2].metric("Rows loaded", f"{len(feed.bars):,}")
source_cols[3].metric("Rows after warm-up", f"{len(bars):,}")
if feed.warning:
    st.warning(feed.warning)

if page == "Sniper Dashboard":
    st.subheader("Sniper Dashboard")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Can trade now?", "YES" if active_plan else "WAIT")
    a2.metric("Final action", veto["final_action"])
    a3.metric("MTF", mtf_confirmation.get("verdict", "INCOMPLETE"))
    a4.metric("Exhaustion", exhaustion_dict.get("state", "clear"))
    st.caption(f"Bull/Bear validation: {bias_validation.get('overall_verdict', 'NO EDGE')} | Side: {bias_validation.get('side', 'NONE')}")
    if veto["reasons"]:
        st.error("Blockers: " + "; ".join(veto["reasons"]))
    if exhaustion_dict.get("warnings"):
        st.warning("; ".join(exhaustion_dict.get("warnings", [])))
    if not active_plan:
        st.warning("No active trade plan. Wait for a clean actionable signal.")
        if st.button("Scan scalp now", type="primary"):
            scalp_display_for_scan = _display_scalp_table(scalp_decision)
            outcome = _show_scalp_scan_result(scalp_decision, scalp_display_for_scan)
            append_scan_log(_scan_log_row(outcome, scalp_decision, scalp_display_for_scan, market, scores, veto, mtf_confirmation, exhaustion_dict))
            st.caption("Scan logged to data/scan_log.csv")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry low", fmt_price(plan_value(display_plan, "entry_zone_low")))
        p2.metric("Entry high", fmt_price(plan_value(display_plan, "entry_zone_high")))
        p3.metric("SL", fmt_price(plan_value(display_plan, "stop")))
        p4.metric("TP1", fmt_price(plan_value(display_plan, "tp1")))
        st.metric("TP2", fmt_price(plan_value(display_plan, "tp2")))

elif page == "Forecast Manager":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final action", veto["final_action"])
    c2.metric("Raw action", action)
    c3.metric("Plan status", status)
    c4.metric("Regime", regime)
    c5.metric("Confidence", scores["confidence"])
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Bull score", scores["bull_score"])
    s2.metric("Bear score", scores["bear_score"])
    s3.metric("Range position", f"{market.range_position_pct:.1f}%")
    s4.metric("Room ratio", f"{veto['room_ratio']:.2f}")
    st.info(summary)
    st.caption(f"Bull/Bear validation: {bias_validation.get('overall_verdict', 'NO EDGE')} | Side: {bias_validation.get('side', 'NONE')}")
    st.caption(f"MTF confirmation: {mtf_confirmation.get('verdict', 'INCOMPLETE')} | Side: {mtf_confirmation.get('side', 'NONE')}")
    st.caption(plan.get("note", ""))
    if candidate_note and (not has_plan_levels(plan) or veto["final_action"] not in ACTIONABLE):
        st.warning(candidate_note)
    if veto["reasons"]:
        st.error("Veto reasons: " + "; ".join(veto["reasons"]))
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Entry zone low", fmt_price(plan_value(display_plan, "entry_zone_low")))
    p2.metric("Entry zone high", fmt_price(plan_value(display_plan, "entry_zone_high")))
    p3.metric("Stop Loss", fmt_price(plan_value(display_plan, "stop")))
    p4.metric("Take Profit 1", fmt_price(plan_value(display_plan, "tp1")))
    p5, p6 = st.columns(2)
    p5.metric("Take Profit 2", fmt_price(plan_value(display_plan, "tp2")))
    p6.metric("Advisory units @ $10k equity", fmt_units(advisory_qty))
    if status != f"Active {veto['final_action']}":
        st.caption("Displayed levels are candidate/preview levels only. Execute manually only when final action is BUY PLAN or SELL PLAN and quality is Clean.")
    if not active_plan:
        st.warning("No active trade plan. Wait for a clean actionable signal.")
        if st.button("Scan scalp now", type="primary"):
            scalp_display_for_scan = _display_scalp_table(scalp_decision)
            outcome = _show_scalp_scan_result(scalp_decision, scalp_display_for_scan)
            append_scan_log(_scan_log_row(outcome, scalp_decision, scalp_display_for_scan, market, scores, veto, mtf_confirmation, exhaustion_dict))
            st.caption("Scan logged to data/scan_log.csv")

elif page == "Bull/Bear Validation":
    st.subheader("Bull/Bear Validation")
    b1, b2, b3 = st.columns(3)
    b1.metric("Current bias", bias_validation.get("bias", "mixed"))
    b2.metric("Validation side", bias_validation.get("side", "NONE"))
    b3.metric("Overall verdict", bias_validation.get("overall_verdict", "NO EDGE"))
    if bias_validation.get("warning"):
        st.warning(bias_validation["warning"])
    validation_display = bias_validation_table(bias_validation)
    main_cols = ["horizon", "side", "status", "verdict", "sample_count", "follow_through_pct", "avg_favorable_move", "avg_adverse_move", "edge_score", "reason"]
    st.dataframe(validation_display[[c for c in main_cols if c in validation_display.columns]], use_container_width=True, hide_index=True)
    with st.expander("Current matched setup"):
        st.json(bias_validation.get("setup", {}))

elif page == "MTF Confirmation":
    st.subheader("Multi-Timeframe Confirmation")
    st.caption("5m trigger, 15m danger check, 1h structure. Use Timeframe = 5m for full confirmation.")
    m1, m2, m3 = st.columns(3)
    m1.metric("MTF verdict", mtf_confirmation.get("verdict", "INCOMPLETE"))
    m2.metric("MTF side", mtf_confirmation.get("side", "NONE"))
    m3.metric("Source timeframe", mtf_confirmation.get("source_interval", "N/A"))
    if mtf_confirmation.get("reasons"):
        st.warning("; ".join(mtf_confirmation.get("reasons", [])))
    mtf_display = multi_timeframe_table(mtf_confirmation)
    mtf_cols = ["timeframe", "role", "status", "bias", "action", "regime", "confidence", "kc_state", "kc_momentum", "rsi", "price", "reason"]
    st.dataframe(mtf_display[[c for c in mtf_cols if c in mtf_display.columns]], use_container_width=True, hide_index=True)

elif page == "Structure Levels":
    st.subheader("Structure Levels")
    cols = st.columns(4)
    cols[0].metric("Support", fmt_price(market.support), market.support_label)
    cols[1].metric("Resistance", fmt_price(market.resistance), market.resistance_label)
    cols[2].metric("Prev day low", fmt_price(market.previous_day_low))
    cols[3].metric("Prev day high", fmt_price(market.previous_day_high))
    rows = [
        {"level": "support", "price": market.support, "source": market.support_label},
        {"level": "resistance", "price": market.resistance, "source": market.resistance_label},
        {"level": "swing_low", "price": market.swing_low, "source": "recent swing"},
        {"level": "swing_high", "price": market.swing_high, "source": "recent swing"},
        {"level": "asia_low", "price": market.asia_low, "source": "session"},
        {"level": "asia_high", "price": market.asia_high, "source": "session"},
        {"level": "london_low", "price": market.london_low, "source": "session"},
        {"level": "london_high", "price": market.london_high, "source": "session"},
        {"level": "round_support", "price": market.round_support, "source": "round number"},
        {"level": "round_resistance", "price": market.round_resistance, "source": "round number"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if market.liquidity_sweep_up or market.liquidity_sweep_down:
        st.warning(f"Liquidity sweep up={market.liquidity_sweep_up}, down={market.liquidity_sweep_down}")

elif page == "Score Breakdown":
    st.subheader("Bull/Bear Score Breakdown")
    st.dataframe(_score_breakdown_table(scores), use_container_width=True, hide_index=True)
    if scores.get("notes"):
        st.info("; ".join(scores.get("notes", [])))

elif page == "Exhaustion Guard":
    st.subheader("Exhaustion Guard")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("State", exhaustion_dict.get("state", "clear"))
    e2.metric("Block BUY", str(exhaustion_dict.get("block_buy", False)))
    e3.metric("Block SELL", str(exhaustion_dict.get("block_sell", False)))
    e4.metric("Range position", fmt_metric(exhaustion_dict.get("range_position_pct"), 1))
    if exhaustion_dict.get("warnings"):
        st.warning("; ".join(exhaustion_dict.get("warnings", [])))
    st.json(exhaustion_dict)

elif page == "Scalp Gate":
    st.subheader("Scalp Gate")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recommendation", scalp_decision.recommendation); m1.caption("Final answer.")
    m2.metric("Side", scalp_decision.side); m2.caption("Allowed direction.")
    m3.metric("Room ratio", fmt_metric(scalp_decision.room_ratio, 2)); m3.caption("Scalp room / scalp risk.")
    m4.metric("RSI", fmt_metric(scalp_decision.rsi, 1)); m4.caption("Overbought/oversold guard.")
    m5.metric("KC momentum", fmt_metric(scalp_decision.kc_momentum, 2)); m5.caption("Negative=down, positive=up.")
    if scalp_decision.reasons:
        st.error("NO SCALP reasons: " + "; ".join(scalp_decision.reasons))
    else:
        st.success("Scalp Gate is open. Use manual execution only; no broker execution is connected.")
    scalp_display = _display_scalp_table(scalp_decision)
    st.subheader("Very short value guide")
    st.dataframe(_scalp_value_guide(scalp_decision, scalp_display), use_container_width=True, hide_index=True)
    scalp_cols = ["horizon", "side", "status", "explanation", "entry", "stop_loss", "take_profit_1", "take_profit_2", "scalp_edge", "tp1_hit_rate_pct", "tp2_hit_rate_pct", "sl_hit_rate_pct", "train_samples", "validation_samples"]
    st.dataframe(scalp_display[[c for c in scalp_cols if c in scalp_display.columns]], use_container_width=True)
    with st.expander("Scalp Gate audit"):
        audit_cols = ["horizon", "side", "tp1_move", "tp2_move", "sl_move", "reason"]
        st.dataframe(scalp_display[[c for c in audit_cols if c in scalp_display.columns]], use_container_width=True)
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan), use_container_width=True)

elif page == "Multi-Horizon Forecast":
    st.subheader("Multi-Horizon Forecast")
    horizon_display = _display_horizon_table(horizon_forecasts)
    main_cols = ["horizon", "bars_ahead", "side", "status", "engine_preferred", "entry", "stop_loss", "take_profit_1", "take_profit_2", "historical_direction_probability_pct", "sample_count"]
    st.dataframe(horizon_display[[c for c in main_cols if c in horizon_display.columns]], use_container_width=True)
    with st.expander("Audit details - actual historical values used"):
        detail_cols = ["horizon", "side", "tp1_move", "tp2_move", "adverse_move", "sample_count", "matching_setup", "reason"]
        st.dataframe(horizon_display[[c for c in detail_cols if c in horizon_display.columns]], use_container_width=True)

elif page == "KC Squeeze":
    st.subheader("KC Squeeze module")
    st.json(kc)
    cols = ["close", "trend_sma", "bb_upper", "bb_lower", "kc_upper", "kc_lower", "squeeze_on", "squeeze_fired", "kc_momentum", "rsi"]
    st.dataframe(bars[[c for c in cols if c in bars.columns]].tail(30), use_container_width=True)
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan), use_container_width=True)

elif page == "Market Map":
    st.subheader("Market location")
    st.json(market.to_dict())
    st.subheader("Expected bands")
    st.json(bands)
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan), use_container_width=True)

elif page == "Macro / News":
    st.subheader("Macro context")
    st.json(macro)
    st.write("Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches. API calendar can be added later.")

elif page == "Settings":
    st.subheader("Active settings")
    st.json(settings)
    st.download_button("Download settings YAML", yaml.safe_dump(settings, sort_keys=False), file_name="active_settings.yaml")

elif page == "Scan History":
    st.subheader("Scan History")
    history = read_scan_log(limit=300)
    if history.empty:
        st.info("No scans logged yet. Click Scan scalp now on Sniper Dashboard or Forecast Manager.")
    else:
        st.dataframe(history, use_container_width=True)
        st.download_button("Download scan history CSV", history.to_csv(index=False), file_name="xauusd_scan_history.csv")

elif page == "Log Snapshot":
    row = {
        "source": feed.source,
        "warning": feed.warning,
        "price": market.price,
        "regime": regime,
        "kc_state": kc["state"],
        "bias": scores["bias"],
        "confidence": scores["confidence"],
        "raw_action": action,
        "final_action": veto["final_action"],
        "plan_status": status,
        "quality": veto["trade_quality"],
        "veto_reasons": "; ".join(veto["reasons"]),
        "bias_validation_side": bias_validation.get("side"),
        "bias_validation_verdict": bias_validation.get("overall_verdict"),
        "mtf_side": mtf_confirmation.get("side"),
        "mtf_verdict": mtf_confirmation.get("verdict"),
        "exhaustion_state": exhaustion_dict.get("state"),
        "support": market.support,
        "support_label": market.support_label,
        "resistance": market.resistance,
        "resistance_label": market.resistance_label,
        "entry_zone_low": plan_value(display_plan, "entry_zone_low"),
        "entry_zone_high": plan_value(display_plan, "entry_zone_high"),
        "stop_loss": plan_value(display_plan, "stop"),
        "take_profit_1": plan_value(display_plan, "tp1"),
        "take_profit_2": plan_value(display_plan, "tp2"),
        "advisory_units_10k": advisory_qty if active_plan else None,
        "scalp_recommendation": scalp_decision.recommendation,
        "scalp_side": scalp_decision.side,
        "scalp_room_ratio": scalp_decision.room_ratio,
        "scalp_reasons": "; ".join(scalp_decision.reasons),
        "candidate_note": candidate_note,
        "plan_note": plan.get("note", ""),
    }
    snap = pd.DataFrame([row])
    st.dataframe(snap, use_container_width=True)
    st.download_button("Download snapshot CSV", snap.to_csv(index=False), file_name="xauusd_forecast_snapshot.csv")
