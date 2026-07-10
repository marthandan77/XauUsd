from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.ai_explainer import explain
from src.data_feed import FeedResult, load_csv, load_live_bars, make_sample_bars
from src.forecast_engine import choose_action, score_forecast
from src.gate_diagnostics import build_gate_breakdown, signal_stage, summarize_gate_blockers
from src.horizon_forecast import build_multi_horizon_forecast
from src.indicators import add_indicators
from src.kc_squeeze_engine import kc_squeeze_summary
from src.learning.controller import backfill_learning_history, refresh_learning_metrics, run_learning_cycle
from src.macro_engine import macro_context
from src.market_map import build_market_map
from src.range_model import price_bands
from src.regime_engine import classify_regime
from src.trade_plan import advisory_position_size, build_trade_plan
from src.veto_engine import apply_veto

ROOT = Path(__file__).resolve().parent
ACTIONABLE = {"BUY PLAN", "SELL PLAN"}
CONTROL_PREFIX = "control_"

PERSISTED_SETTING_KEYS = [
    "price_interval",
    "price_period",
    "buy_threshold",
    "sell_threshold",
    "wait_threshold",
    "atr_multiplier",
    "min_reward_risk",
    "middle_range_lower_pct",
    "middle_range_upper_pct",
    "kc_squeeze_enabled",
    "bb_length",
    "bb_mult",
    "kc_length",
    "kc_mult",
    "kc_release_recent_bars",
    "kc_release_chase_atr_limit",
    "trend_length",
    "atr_period",
    "atr_stop_multiplier",
    "atr_tp_multiplier",
    "buy_tp1_atr_multiplier",
    "buy_tp2_atr_multiplier",
    "sell_tp1_atr_multiplier",
    "sell_tp2_atr_multiplier",
    "sell_support_buffer_atr",
    "min_tp1_atr_distance",
    "risk_per_trade_pct",
    "show_bollinger_bands",
    "show_keltner_channels",
    "news_block_manual",
    "require_horizon_alignment",
    "horizon_model_lookback_bars",
    "horizon_ridge_alpha",
    "horizon_min_probability_gap",
    "filter_min_probability_gap",
    "horizon_cost_points",
    "walk_forward_tests",
    "walk_forward_min_train",
    "walk_forward_min_tests",
    "walk_forward_min_side_samples",
    "walk_forward_min_accuracy",
    "learning_backfill_points",
    "stale_candle_multiplier",
]

st.set_page_config(page_title="XAU/USD Forecast Manager", layout="wide")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sample_freq(settings: dict) -> str:
    return {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }.get(str(settings.get("price_interval", "5m")), "5min")


def _control_key(key: str) -> str:
    return f"{CONTROL_PREFIX}{key}"


def _query_params_as_dict() -> dict[str, str]:
    try:
        return {key: value for key, value in st.query_params.items()}
    except Exception:
        try:
            raw = st.experimental_get_query_params()
            return {key: values[0] for key, values in raw.items() if values}
        except Exception:
            return {}


def _coerce_query_value(raw_value, current_value):
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else None
    if raw_value is None:
        return current_value
    if isinstance(current_value, bool):
        return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        try:
            return int(float(str(raw_value)))
        except Exception:
            return current_value
    if isinstance(current_value, float):
        try:
            return float(str(raw_value))
        except Exception:
            return current_value
    return str(raw_value)


def apply_query_settings(settings: dict) -> dict:
    updated = dict(settings)
    params = _query_params_as_dict()
    for key in PERSISTED_SETTING_KEYS:
        if key in updated and key in params:
            updated[key] = _coerce_query_value(params[key], updated[key])
    return updated


def _query_serialized_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def persist_query_settings(settings: dict) -> None:
    params = {key: _query_serialized_value(settings[key]) for key in PERSISTED_SETTING_KEYS if key in settings}
    try:
        st.query_params.clear()
        for key, value in params.items():
            st.query_params[key] = value
    except Exception:
        try:
            st.experimental_set_query_params(**params)
        except Exception:
            pass


def clear_query_settings() -> None:
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def initialize_control_state(settings: dict) -> None:
    for key in PERSISTED_SETTING_KEYS:
        if key in settings and _control_key(key) not in st.session_state:
            st.session_state[_control_key(key)] = settings[key]


def clear_control_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(CONTROL_PREFIX):
            del st.session_state[key]
    for key in [
        "last_scan_result",
        "last_scan_error",
        "last_scan_time",
        "learning_metrics",
        "learning_backfill",
    ]:
        st.session_state.pop(key, None)


def staged_settings(settings: dict) -> dict:
    staged = dict(settings)
    for key in PERSISTED_SETTING_KEYS:
        control_key = _control_key(key)
        if control_key in st.session_state:
            staged[key] = st.session_state[control_key]
    return staged


def stage_preset_values(preset: dict) -> None:
    for key, value in preset.items():
        if key in PERSISTED_SETTING_KEYS:
            st.session_state[_control_key(key)] = value


def fmt_price(value) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:,.2f}"


def fmt_units(value) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(number) or number <= 0:
        return "N/A"
    return f"{number:,.2f}"


def fmt_number(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def has_plan_levels(plan: dict) -> bool:
    return plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None


def plan_value(plan: dict, key: str):
    if not has_plan_levels(plan):
        return None
    return plan.get(key)


def candidate_plan_from_scores(scores: dict, market, settings: dict) -> tuple[str, dict, str]:
    empty = {
        "entry_zone_low": None,
        "entry_zone_high": None,
        "stop": None,
        "tp1": None,
        "tp2": None,
        "risk": 0.0,
    }
    if bool(settings.get("signals_disabled", False)):
        empty["note"] = "Signals disabled until real data is active."
        return "NO CANDIDATE", empty, ""

    if bool(scores.get("force_wait", False)):
        empty["note"] = scores.get("wait_reason", "Waiting for confirmation.")
        return "NO CANDIDATE", empty, ""

    threshold = int(settings.get("forecast_threshold", 65))
    bias = str(scores.get("bias", "mixed"))
    bull_score = int(scores.get("bull_score", 0))
    bear_score = int(scores.get("bear_score", 0))

    if bias == "bullish" and bull_score >= threshold and bool(settings.get("long_plans_enabled", True)):
        plan = build_trade_plan("BUY PLAN", market, settings)
        plan["note"] = "Bullish preview only. Use only if final action becomes BUY PLAN and veto filters are clean."
        return "BUY CANDIDATE", plan, "Buyer pressure is developing, but this is preview only unless final action is BUY PLAN."

    if bias == "bearish" and bear_score >= threshold:
        preview_settings = dict(settings)
        preview_settings["short_plans_enabled"] = True
        plan = build_trade_plan("SELL PLAN", market, preview_settings)
        if bool(settings.get("short_plans_enabled", False)):
            plan["note"] = "Bearish preview only. Use only if final action becomes SELL PLAN and veto filters are clean."
            return "SELL CANDIDATE", plan, "Seller pressure is developing, but this is preview only unless final action is SELL PLAN."
        plan["note"] = "Bearish preview only. Short plans are disabled, so this is not an active SELL PLAN."
        return "SELL PREVIEW ONLY", plan, "Seller pressure is developing, but this is not an active sell signal."

    empty["note"] = "No active trade plan. Wait for a cleaner case."
    return "NO CANDIDATE", empty, ""


def plan_status(action: str, final_action: str, candidate_action: str) -> str:
    if final_action == "BUY PLAN":
        return "Buy allowed"
    if final_action == "SELL PLAN":
        return "Sell allowed"
    if candidate_action == "BUY CANDIDATE":
        return "Buyer pressure only"
    if candidate_action in {"SELL CANDIDATE", "SELL PREVIEW ONLY"}:
        return "Seller pressure only"
    if action in ACTIONABLE:
        return f"Rejected {action}"
    return "No active trade"


def render_sidebar_backfill() -> None:
    result = st.session_state.get("last_scan_result")
    ready = isinstance(result, dict) and bool(result.get("ok")) and result.get("bars") is not None
    clicked = st.sidebar.button(
        "Backfill Learning History",
        key="learning_backfill_sidebar_button",
        disabled=not ready,
        use_container_width=True,
        help="Run SCAN NOW first. Backfill uses the latest scanned bars to create historical walk-forward outcome samples.",
    )
    st.sidebar.caption("Backfill is enabled only after SCAN NOW. It is for diagnosis, not auto-optimization.")
    if clicked:
        with st.sidebar:
            with st.spinner("Backfilling learning history..."):
                status = backfill_learning_history(result)
        st.session_state["learning_backfill"] = status
        st.session_state["learning_metrics"] = refresh_learning_metrics()
        if status.get("ok"):
            st.sidebar.success(f"Backfilled {status.get('outcome_count', 0)} outcomes.")
        else:
            st.sidebar.warning(status.get("reason", "Backfill failed."))


def settings_panel(settings: dict, presets: dict) -> dict:
    initialize_control_state(settings)

    left_col, right_col = st.sidebar.columns(2)
    reset_clicked = left_col.button("Reset", key="settings_reset_button", use_container_width=True)
    apply_clicked = right_col.button("Apply", key="settings_apply_button", type="primary", use_container_width=True)
    render_sidebar_backfill()

    if reset_clicked:
        clear_query_settings()
        clear_control_state()
        st.rerun()

    st.sidebar.caption("Change controls first, press Apply, then press SCAN NOW to recalculate.")
    st.sidebar.header("Forecast controls")

    if presets:
        selected = st.sidebar.selectbox("Preset", list(presets.keys()), index=0, key="preset_selector")
        if st.sidebar.button("Stage preset", key="stage_preset_button"):
            stage_preset_values(presets[selected])
            st.rerun()

    interval_options = ["5m", "15m", "30m", "1h", "4h", "1d"]
    if st.session_state.get(_control_key("price_interval")) not in interval_options:
        st.session_state[_control_key("price_interval")] = str(settings.get("price_interval", "5m"))
    period_options = ["7d", "14d", "30d", "60d", "6mo", "1y", "2y"]
    if st.session_state.get(_control_key("price_period")) not in period_options:
        st.session_state[_control_key("price_period")] = str(settings.get("price_period", "30d"))

    st.sidebar.selectbox("Timeframe", interval_options, key=_control_key("price_interval"))
    st.sidebar.selectbox("Lookback period", period_options, key=_control_key("price_period"))
    st.sidebar.slider("Bull threshold", 50, 95, key=_control_key("buy_threshold"))
    st.sidebar.slider("Bear threshold", 50, 95, key=_control_key("sell_threshold"))
    st.sidebar.slider("Watch threshold", 40, 90, key=_control_key("wait_threshold"))
    st.sidebar.slider("Minimum room ratio", 1.0, 3.0, step=0.1, key=_control_key("min_reward_risk"))
    st.sidebar.slider("Middle zone lower %", 10, 49, key=_control_key("middle_range_lower_pct"))
    st.sidebar.slider("Middle zone upper %", 51, 90, key=_control_key("middle_range_upper_pct"))

    st.sidebar.header("KC Squeeze")
    st.sidebar.toggle("Enable KC Squeeze module", key=_control_key("kc_squeeze_enabled"))
    st.sidebar.slider("BB length", 10, 60, key=_control_key("bb_length"))
    st.sidebar.slider("BB multiplier", 1.0, 3.5, step=0.1, key=_control_key("bb_mult"))
    st.sidebar.slider("KC length", 10, 60, key=_control_key("kc_length"))
    st.sidebar.slider("KC multiplier", 1.0, 3.5, step=0.1, key=_control_key("kc_mult"))
    st.sidebar.slider("KC release memory bars", 1, 8, key=_control_key("kc_release_recent_bars"))
    st.sidebar.slider("KC chase limit ATR", 0.5, 3.0, step=0.1, key=_control_key("kc_release_chase_atr_limit"))
    st.sidebar.slider("Trend SMA length", 50, 300, key=_control_key("trend_length"))
    st.sidebar.slider("ATR period", 5, 50, key=_control_key("atr_period"))

    st.sidebar.header("Trade plan")
    st.sidebar.slider("ATR stop multiplier", 0.8, 5.0, step=0.1, key=_control_key("atr_stop_multiplier"))
    st.sidebar.slider("Buy Take Profit 1 ATR", 0.50, 5.00, step=0.05, key=_control_key("buy_tp1_atr_multiplier"))
    st.sidebar.slider("Buy Take Profit 2 ATR", 0.75, 8.00, step=0.05, key=_control_key("buy_tp2_atr_multiplier"))
    st.sidebar.slider("Sell Take Profit 1 ATR", 0.50, 5.00, step=0.05, key=_control_key("sell_tp1_atr_multiplier"))
    st.sidebar.slider("Sell Take Profit 2 ATR", 0.75, 8.00, step=0.05, key=_control_key("sell_tp2_atr_multiplier"))
    st.sidebar.slider("Minimum TP1 distance ATR", 0.25, 3.00, step=0.05, key=_control_key("min_tp1_atr_distance"))
    st.sidebar.slider("Sell support buffer ATR", 0.00, 1.00, step=0.05, key=_control_key("sell_support_buffer_atr"))
    st.sidebar.slider("Advisory risk %", 0.1, 2.0, step=0.1, key=_control_key("risk_per_trade_pct"))

    st.sidebar.header("Horizon gate")
    st.sidebar.toggle("Require horizon alignment", key=_control_key("require_horizon_alignment"))
    st.sidebar.slider("Model lookback bars", 500, 5000, step=100, key=_control_key("horizon_model_lookback_bars"))
    st.sidebar.slider("Ridge alpha", 0.1, 10.0, step=0.1, key=_control_key("horizon_ridge_alpha"))
    st.sidebar.slider("5m probability gap", 0.01, 0.20, step=0.01, key=_control_key("horizon_min_probability_gap"))
    st.sidebar.slider("Filter probability gap", 0.01, 0.25, step=0.01, key=_control_key("filter_min_probability_gap"))
    st.sidebar.slider("Assumed cost points", 0.0, 3.0, step=0.05, key=_control_key("horizon_cost_points"))
    st.sidebar.slider("Walk-forward tests", 20, 200, step=10, key=_control_key("walk_forward_tests"))
    st.sidebar.slider("Walk-forward min train", 120, 1000, step=20, key=_control_key("walk_forward_min_train"))
    st.sidebar.slider("Walk-forward min tests", 10, 120, step=5, key=_control_key("walk_forward_min_tests"))
    st.sidebar.slider("Walk-forward side samples", 3, 50, step=1, key=_control_key("walk_forward_min_side_samples"))
    st.sidebar.slider("Walk-forward min accuracy", 50.0, 75.0, step=1.0, key=_control_key("walk_forward_min_accuracy"))
    st.sidebar.slider("Backfill points / horizon", 5, 250, step=5, key=_control_key("learning_backfill_points"))
    st.sidebar.slider("Stale candle multiplier", 1.0, 5.0, step=0.5, key=_control_key("stale_candle_multiplier"))

    st.sidebar.caption("BUY and SELL plan processing is always enabled.")
    st.sidebar.toggle("Show Bollinger Bands", key=_control_key("show_bollinger_bands"))
    st.sidebar.toggle("Show Keltner Channels", key=_control_key("show_keltner_channels"))
    st.sidebar.toggle("Manual event block", key=_control_key("news_block_manual"))

    if apply_clicked:
        applied = staged_settings(settings)
        applied["long_plans_enabled"] = True
        applied["short_plans_enabled"] = True
        persist_query_settings(applied)
        st.sidebar.success("Settings applied. Press SCAN NOW to recalculate.")
        return applied

    settings["long_plans_enabled"] = True
    settings["short_plans_enabled"] = True
    return settings


def refresh_panel() -> None:
    st.sidebar.header("Refresh")
    if st.sidebar.button("Manual refresh page", type="secondary"):
        st.rerun()
    st.sidebar.caption("Manual page refresh does not rescan. Use SCAN NOW on Forecast Manager to recalculate.")


def data_panel(settings: dict) -> dict:
    st.sidebar.header("Data")
    data_mode = st.sidebar.radio("Data mode", ["Twelve Data API", "CSV upload", "Sample"], index=0, key="data_mode")
    settings["data_mode"] = data_mode
    settings["uploaded_file"] = None
    if data_mode == "CSV upload":
        settings["uploaded_file"] = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
    elif data_mode == "Twelve Data API":
        key_input = st.sidebar.text_input(
            "Twelve Data API key - optional local session override",
            value="",
            type="password",
            help="Use this only if Streamlit cannot see your environment variable. The key is not saved to GitHub.",
            key="twelve_data_runtime_key",
        )
        if key_input.strip():
            settings["twelve_data_api_key_runtime"] = key_input.strip()
    return settings


def fetch_bars(settings: dict) -> FeedResult:
    data_mode = str(settings.get("data_mode", "Twelve Data API"))
    if data_mode == "CSV upload":
        uploaded = settings.get("uploaded_file")
        if uploaded is not None:
            return load_csv(uploaded)
        return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "CSV not uploaded; sample data used")
    if data_mode == "Twelve Data API":
        feed = load_live_bars(settings)
        if not feed.bars.empty:
            return feed
        return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "Twelve Data unavailable; sample data used. " + feed.warning)
    return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "sample data")


def safe_settings(settings: dict) -> dict:
    return {key: value for key, value in settings.items() if key != "uploaded_file"}


def run_strategy_scan(settings: dict, macro_bias: str) -> dict:
    scan_settings = dict(settings)
    feed = fetch_bars(scan_settings)
    scan_settings["signals_disabled"] = feed.source == "sample"
    scan_settings["signals_disabled_reason"] = "Sample data source; trade signals are disabled until live API or uploaded real CSV is active."

    bars_raw = add_indicators(feed.bars, scan_settings)
    bars = bars_raw.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
    if bars.empty:
        return {
            "ok": False,
            "error": "Not enough clean OHLC data after indicator warm-up. Increase lookback period or use real CSV data.",
            "feed": feed,
            "settings": safe_settings(scan_settings),
            "scan_time": pd.Timestamp.now(tz="Asia/Singapore"),
            "multi_horizon": [],
        }

    multi_horizon = build_multi_horizon_forecast(bars, scan_settings)
    kc = kc_squeeze_summary(bars, scan_settings)
    latest_row = bars.iloc[-1].to_dict()
    latest_row["kc_state"] = kc["state"]
    latest_row["kc_reason"] = kc["reason"]

    market = build_market_map(bars, scan_settings)
    regime = classify_regime(bars, market, scan_settings)
    macro = macro_context(macro_bias, bool(scan_settings.get("news_block_manual", False)))
    scores = score_forecast(latest_row, market, regime, macro, scan_settings)
    if kc["state"] != "insufficient_data":
        scores["kc_state"] = kc["state"]
        scores["kc_reason"] = kc["reason"]

    action = choose_action(scores, scan_settings)
    plan = build_trade_plan(action, market, scan_settings)
    candidate_action, candidate_plan, candidate_note = candidate_plan_from_scores(scores, market, scan_settings)
    bands = price_bands(market, regime)
    veto = apply_veto(action, plan, market, regime, macro, scan_settings, latest_row, multi_horizon)
    summary = explain(regime, scores, veto, market, macro)
    active_plan = veto["final_action"] in ACTIONABLE
    status = plan_status(action, veto["final_action"], candidate_action)
    display_plan = plan if has_plan_levels(plan) else candidate_plan

    advisory_qty = 0.0
    if active_plan and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None:
        advisory_qty = advisory_position_size(
            account_equity=10000,
            risk_pct=float(scan_settings.get("risk_per_trade_pct", 0.5)) / 100.0,
            entry=(float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0,
            stop=float(plan["stop"]),
        )

    result = {
        "ok": True,
        "scan_time": pd.Timestamp.now(tz="Asia/Singapore"),
        "feed": feed,
        "settings": safe_settings(scan_settings),
        "bars": bars,
        "kc": kc,
        "latest_row": latest_row,
        "market": market,
        "regime": regime,
        "macro": macro,
        "scores": scores,
        "action": action,
        "plan": plan,
        "candidate_action": candidate_action,
        "candidate_plan": candidate_plan,
        "candidate_note": candidate_note,
        "bands": bands,
        "veto": veto,
        "summary": summary,
        "active_plan": active_plan,
        "status": status,
        "display_plan": display_plan,
        "advisory_qty": advisory_qty,
        "multi_horizon": multi_horizon,
    }
    result["signal_stage"] = signal_stage(result)
    result["gate_breakdown"] = build_gate_breakdown(result)
    result["gate_summary"] = summarize_gate_blockers(result)
    return result


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
    if "squeeze_fired" in chart_df:
        release_df = chart_df[chart_df["squeeze_fired"].fillna(False).astype(bool)]
        if not release_df.empty:
            fig.add_trace(go.Scatter(x=release_df.index, y=release_df["close"], mode="markers", name="KC Release"))
    if action in ACTIONABLE and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None:
        marker_price = (float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0
        fig.add_trace(go.Scatter(x=[chart_df.index[-1]], y=[marker_price], mode="markers+text", text=[action], textposition="top center", name="Advisory signal"))
    fig.update_layout(height=640, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render_no_scan() -> None:
    st.info("No scan result yet. Press SCAN NOW on Forecast Manager to fetch data, calculate indicators, and run the strategy.")


def render_source_metrics(result: dict) -> None:
    feed = result["feed"]
    bars = result.get("bars")
    market = result.get("market")
    cols = st.columns(5)
    cols[0].metric("Data source", feed.source)
    cols[1].metric("Latest price", fmt_price(market.price if market is not None else None))
    cols[2].metric("Rows loaded", f"{len(feed.bars):,}")
    cols[3].metric("Rows after warm-up", f"{len(bars):,}" if bars is not None else "0")
    cols[4].metric("Last scan", result["scan_time"].strftime("%H:%M:%S"))
    if feed.warning:
        st.warning(feed.warning)
    if bool(result["settings"].get("signals_disabled", False)):
        st.error(result["settings"].get("signals_disabled_reason", "Signals disabled."))


def score_label(score: int, threshold: int) -> str:
    if score >= threshold:
        return "strong enough for a trade check"
    if score >= 65:
        return "medium pressure, watch only"
    if score >= 50:
        return "developing pressure"
    return "weak pressure"


def regime_label(regime: str) -> str:
    return {
        "range": "sideways market",
        "bull_trend": "buyers controlling the trend",
        "bear_trend": "sellers controlling the trend",
        "compression": "market storing energy before a breakout",
        "squeeze_release_now": "KC compression just released, direction not confirmed",
        "squeeze_release_recent": "recent KC release, direction not confirmed",
        "squeeze_release_expired": "previous KC release expired",
        "bullish_release_confirmed": "upside release confirmed",
        "bearish_release_confirmed": "downside release confirmed",
        "bullish_release_overextended": "upside release is already stretched",
        "bearish_release_overextended": "downside release is already stretched",
        "shock": "abnormal volatility, avoid new trades",
    }.get(regime, regime.replace("_", " "))


def price_location_label(range_position: float) -> str:
    if range_position <= 30:
        return "near support / lower part of the range"
    if range_position >= 70:
        return "near resistance / upper part of the range"
    return "middle of the range"


def room_label(room_ratio: float) -> str:
    if room_ratio >= 1.5:
        return "good target room"
    if room_ratio >= 1.2:
        return "acceptable target room"
    if room_ratio >= 0.8:
        return "borderline target room"
    return "poor target room"


def setup_label(result: dict) -> str:
    final_action = result["veto"]["final_action"]
    candidate = result["candidate_action"]
    if final_action == "BUY PLAN":
        return "Buy setup is actionable."
    if final_action == "SELL PLAN":
        return "Sell setup is actionable."
    if candidate == "BUY CANDIDATE":
        return "Buyer pressure exists, but it is not an active buy signal."
    if candidate in {"SELL CANDIDATE", "SELL PREVIEW ONLY"}:
        return "Seller pressure exists, but it is not an active sell signal."
    return "No clear trade setup yet."


def build_simple_market_case(result: dict) -> dict:
    scores = result["scores"]
    market = result["market"]
    veto = result["veto"]
    settings = result["settings"]
    bull = int(scores.get("bull_score", 0))
    bear = int(scores.get("bear_score", 0))
    buy_threshold = int(settings.get("buy_threshold", 78))
    sell_threshold = int(settings.get("sell_threshold", 78))
    room_ratio = float(veto.get("room_ratio", 0.0) or 0.0)
    range_position = float(market.range_position_pct)
    final_action = str(veto.get("final_action", "HOLD"))
    stage = result.get("signal_stage") or signal_stage(result)

    side = "buyers" if bull > bear else "sellers" if bear > bull else "neither side"
    strength = max(bull, bear)
    strength_text = score_label(strength, buy_threshold if bull >= bear else sell_threshold)
    market_type = regime_label(result["regime"])
    location = price_location_label(range_position)
    room = room_label(room_ratio)

    if final_action == "BUY PLAN":
        permission = "BUY ALLOWED"
        permission_detail = "The buy setup passed the current trade checks."
    elif final_action == "SELL PLAN":
        permission = "SELL ALLOWED"
        permission_detail = "The sell setup passed the current trade checks."
    elif final_action == "WAIT":
        permission = "WAIT — NO TRADE"
        permission_detail = "There is pressure in one direction, but the setup is not clean enough to trade."
    else:
        permission = "HOLD — NO EDGE"
        permission_detail = "The market does not show a clean trading edge right now."

    case_lines = [
        f"Signal stage: {stage}.",
        f"Gold is in a {market_type}.",
        f"Current advantage: {side} with {strength_text}.",
        f"Price location: {location} at {range_position:.1f}% of the recent range.",
        f"Target room: {room} with room ratio {room_ratio:.2f}.",
        setup_label(result),
    ]

    why_lines = []
    if bear > bull:
        why_lines.append(f"Seller strength is {bear}/100. Sell trigger needs {sell_threshold}/100.")
    elif bull > bear:
        why_lines.append(f"Buyer strength is {bull}/100. Buy trigger needs {buy_threshold}/100.")
    else:
        why_lines.append(f"Buyer and seller strength are balanced at {bull}/100.")
    why_lines.append(f"Buyer strength: {bull}/100. Seller strength: {bear}/100. These are signal-strength scores, not win probabilities.")
    why_lines.append(f"Price is {location}. Range position is {range_position:.1f}%.")
    why_lines.append(f"Target room is {room}. Room ratio is {room_ratio:.2f}.")
    if veto.get("reasons"):
        why_lines.extend([f"Blocked by: {reason}." for reason in veto["reasons"]])
    elif final_action not in ACTIONABLE:
        why_lines.append("No veto rejection, but score/location/room are not strong enough for an active trade.")

    return {
        "permission": permission,
        "permission_detail": permission_detail,
        "case_lines": case_lines,
        "why_lines": why_lines,
        "buyer_strength": bull,
        "seller_strength": bear,
    }


def render_gate_breakdown(result: dict) -> None:
    rows = result.get("gate_breakdown") or build_gate_breakdown(result)
    st.subheader("Gate Breakdown")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("This table shows exactly which module is blocking, watching, or passing. Backfill does not change these gates automatically.")


def render_simple_case_panel(result: dict) -> None:
    case = build_simple_market_case(result)
    st.subheader("Actual market case")
    st.info("\n\n".join(case["case_lines"]))

    st.subheader("Trade permission")
    if result["veto"]["final_action"] in ACTIONABLE:
        st.success(f"{case['permission']} — {case['permission_detail']}")
    else:
        st.warning(f"{case['permission']} — {case['permission_detail']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Buyer strength", f"{case['buyer_strength']}/100")
    c2.metric("Seller strength", f"{case['seller_strength']}/100")
    c3.metric("Signal stage", result.get("signal_stage") or signal_stage(result))
    st.caption("Strength score is not win probability. It is the engine's raw signal pressure score.")

    render_gate_breakdown(result)

    st.subheader("Why")
    for line in case["why_lines"]:
        st.write(f"- {line}")


def _horizon_rows(forecasts: list[dict]) -> list[dict]:
    rows = []
    for item in forecasts or []:
        rows.append({
            "Timeframe": item.get("horizon"),
            "Purpose": "Entry" if item.get("role") == "Signal" else "Filter",
            "Current Read": item.get("bias"),
            "Bull Accuracy": item.get("bull_confidence"),
            "Bear Accuracy": item.get("bear_confidence"),
            "Validation": item.get("validation_status"),
            "Tests": item.get("validation_tests"),
            "Decision": item.get("decision"),
        })
    return rows


def _horizon_advanced_rows(forecasts: list[dict]) -> list[dict]:
    rows = []
    for item in forecasts or []:
        rows.append({
            "Horizon": item.get("horizon"),
            "Role": item.get("role"),
            "Bias": item.get("bias"),
            "Model Up %": item.get("model_up_probability"),
            "Model Down %": item.get("model_down_probability"),
            "EV buy": item.get("ev_buy_points"),
            "EV sell": item.get("ev_sell_points"),
            "Min EV": item.get("min_ev_points"),
            "Active Accuracy": item.get("active_accuracy"),
            "Active Samples": item.get("active_samples"),
            "MAE points": item.get("mae_points"),
            "Expected low": item.get("expected_low"),
            "Expected high": item.get("expected_high"),
            "Training rows": item.get("training_rows"),
            "Status": item.get("status"),
        })
    return rows


def render_multi_horizon_panel(result: dict) -> None:
    st.subheader("5m Entry + Higher-Timeframe Filters")
    forecasts = result.get("multi_horizon") or []
    if not forecasts:
        st.info("Multi-horizon output is unavailable for this scan.")
        return
    st.dataframe(pd.DataFrame(_horizon_rows(forecasts)), use_container_width=True, hide_index=True)
    st.caption("Current Read = model read now. Bull/Bear Accuracy = recent walk-forward hit-rate for that side. 5m is entry; 1h, 4h, and Daily are filters only.")
    with st.expander("Advanced horizon model details"):
        st.dataframe(pd.DataFrame(_horizon_advanced_rows(forecasts)), use_container_width=True, hide_index=True)


def render_trade_levels(result: dict) -> None:
    veto = result["veto"]
    display_plan = result["display_plan"]
    if veto["final_action"] not in ACTIONABLE:
        st.subheader("Trade levels")
        st.write("No active trade plan. Candidate levels are hidden to avoid accidental trading.")
        if result["candidate_action"] != "NO CANDIDATE":
            with st.expander("Preview levels only — not a signal"):
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Entry low", fmt_price(plan_value(display_plan, "entry_zone_low")))
                p2.metric("Entry high", fmt_price(plan_value(display_plan, "entry_zone_high")))
                p3.metric("Stop", fmt_price(plan_value(display_plan, "stop")))
                p4.metric("TP1", fmt_price(plan_value(display_plan, "tp1")))
                p5, p6 = st.columns(2)
                p5.metric("TP2", fmt_price(plan_value(display_plan, "tp2")))
                p6.metric("TP1 distance ATR", fmt_number(display_plan.get("tp1_distance_atr"), 2))
        return

    st.subheader("Active trade levels")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Entry low", fmt_price(plan_value(display_plan, "entry_zone_low")))
    p2.metric("Entry high", fmt_price(plan_value(display_plan, "entry_zone_high")))
    p3.metric("Stop Loss", fmt_price(plan_value(display_plan, "stop")))
    p4.metric("Take Profit 1", fmt_price(plan_value(display_plan, "tp1")))
    p5, p6, p7 = st.columns(3)
    p5.metric("Take Profit 2", fmt_price(plan_value(display_plan, "tp2")))
    p6.metric("TP1 distance ATR", fmt_number(display_plan.get("tp1_distance_atr"), 2))
    p7.metric("Advisory units @ $10k equity", fmt_units(result["advisory_qty"]))


def render_advanced_details(result: dict) -> None:
    scores = result["scores"]
    veto = result["veto"]
    market = result["market"]
    with st.expander("Advanced details"):
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Final action", veto["final_action"])
        a2.metric("Raw action", result["action"])
        a3.metric("Setup status", result["status"])
        a4.metric("Regime", regime_label(result["regime"]))
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Bull score", scores["bull_score"])
        b2.metric("Bear score", scores["bear_score"])
        b3.metric("Range position", f"{market.range_position_pct:.1f}%")
        b4.metric("Room ratio", f"{veto['room_ratio']:.2f}")
        st.write("Engine summary:")
        st.write(result["summary"])
        st.write("Gate summary:")
        st.write(result.get("gate_summary") or summarize_gate_blockers(result))
        if scores.get("force_wait") and scores.get("wait_reason"):
            st.warning(scores["wait_reason"])
        if result["candidate_note"]:
            st.caption(result["candidate_note"])
        if veto["reasons"]:
            st.error("Veto reasons: " + "; ".join(veto["reasons"]))


def render_forecast_manager(result: dict) -> None:
    render_source_metrics(result)
    if not result.get("ok"):
        st.error(result.get("error", "Scan failed."))
        return
    render_simple_case_panel(result)
    render_multi_horizon_panel(result)
    render_trade_levels(result)
    render_advanced_details(result)
    st.plotly_chart(price_chart(result["bars"], result["settings"], result["veto"]["final_action"], result["plan"]), use_container_width=True)


def render_kc_page(result: dict) -> None:
    if not result.get("ok"):
        st.error(result.get("error", "Scan failed."))
        return
    kc = result["kc"]
    bars = result["bars"]
    st.subheader("KC Squeeze module")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("KC state", kc["state"])
    k2.metric("Direction", kc.get("release_direction", "none"))
    k3.metric("Squeeze on", str(kc.get("squeeze_on", False)))
    k4.metric("Release now", str(kc.get("squeeze_fired", False)))
    k5.metric("Bars since release", fmt_number(kc.get("bars_since_squeeze_release"), 0))
    k6.metric("Chase ATR", fmt_number(kc.get("release_chase_atr"), 2))
    st.json(kc)
    cols = [
        "close", "trend_sma", "bb_upper", "bb_lower", "kc_upper", "kc_lower", "compression_ratio", "squeeze_on",
        "squeeze_fired", "squeeze_recent", "bars_since_squeeze_release", "release_direction", "release_chase_atr",
        "release_bullish_confirmed", "release_bearish_confirmed", "kc_momentum",
    ]
    st.dataframe(bars[[c for c in cols if c in bars.columns]].tail(30), use_container_width=True)
    st.plotly_chart(price_chart(bars, result["settings"], result["veto"]["final_action"], result["plan"]), use_container_width=True)


def render_market_map_page(result: dict) -> None:
    if not result.get("ok"):
        st.error(result.get("error", "Scan failed."))
        return
    st.subheader("Market location")
    st.json(result["market"].to_dict())
    st.subheader("Expected bands")
    st.json(result["bands"])
    st.plotly_chart(price_chart(result["bars"], result["settings"], result["veto"]["final_action"], result["plan"]), use_container_width=True)


def render_log_snapshot(result: dict) -> None:
    if not result.get("ok"):
        st.error(result.get("error", "Scan failed."))
        return
    row = {
        "scan_time": result["scan_time"].isoformat(),
        "source": result["feed"].source,
        "warning": result["feed"].warning,
        "signals_disabled": result["settings"].get("signals_disabled", False),
        "price": result["market"].price,
        "signal_stage": result.get("signal_stage"),
        "gate_summary": result.get("gate_summary"),
        "regime": result["regime"],
        "kc_state": result["kc"]["state"],
        "kc_direction": result["kc"].get("release_direction"),
        "bias": result["scores"]["bias"],
        "rule_score": result["scores"].get("rule_score", result["scores"].get("confidence")),
        "raw_action": result["action"],
        "final_action": result["veto"]["final_action"],
        "plan_status": result["status"],
        "quality": result["veto"]["trade_quality"],
        "veto_reasons": "; ".join(result["veto"]["reasons"]),
        "entry_zone_low": plan_value(result["display_plan"], "entry_zone_low"),
        "entry_zone_high": plan_value(result["display_plan"], "entry_zone_high"),
        "stop_loss": plan_value(result["display_plan"], "stop"),
        "take_profit_1": plan_value(result["display_plan"], "tp1"),
        "take_profit_2": plan_value(result["display_plan"], "tp2"),
        "tp1_distance_atr": result["display_plan"].get("tp1_distance_atr"),
        "tp1_too_close": result["display_plan"].get("tp1_too_close"),
        "advisory_units_10k": result["advisory_qty"] if result["active_plan"] else None,
        "candidate_note": result["candidate_note"],
        "plan_note": result["plan"].get("note", ""),
    }
    snap = pd.DataFrame([row])
    horizon = pd.DataFrame(_horizon_rows(result.get("multi_horizon", [])))
    gates = pd.DataFrame(result.get("gate_breakdown") or build_gate_breakdown(result))
    st.subheader("Scan snapshot")
    st.dataframe(snap, use_container_width=True)
    st.download_button("Download snapshot CSV", snap.to_csv(index=False), file_name="xauusd_forecast_snapshot.csv")
    st.subheader("Gate snapshot")
    st.dataframe(gates, use_container_width=True, hide_index=True)
    st.download_button("Download gate CSV", gates.to_csv(index=False), file_name="xauusd_gate_snapshot.csv")
    st.subheader("Horizon snapshot")
    st.dataframe(horizon, use_container_width=True, hide_index=True)
    st.download_button("Download horizon CSV", horizon.to_csv(index=False), file_name="xauusd_horizon_snapshot.csv")


def render_learning_page(result: dict | None) -> None:
    st.subheader("GitHub Learning")
    st.write("Learning is GitHub-backed. It logs predictions, evaluates expired outcomes, and reads backfill batches. It does not generate recommended settings or auto-tune live settings.")

    if result is not None and result.get("learning"):
        st.subheader("Last learning cycle")
        st.json(result.get("learning"))

    col1, col2 = st.columns(2)
    if col1.button("Refresh learning metrics", use_container_width=True):
        with st.spinner("Reading learning outcomes from GitHub..."):
            st.session_state["learning_metrics"] = refresh_learning_metrics()
    if col2.button("Backfill from last scan", use_container_width=True, disabled=result is None or not result.get("ok", False)):
        with st.spinner("Backfilling learning history from latest scan bars..."):
            st.session_state["learning_backfill"] = backfill_learning_history(result)
            st.session_state["learning_metrics"] = refresh_learning_metrics()

    if st.session_state.get("learning_backfill"):
        st.subheader("Last backfill")
        st.json(st.session_state["learning_backfill"])

    metrics_payload = st.session_state.get("learning_metrics") or (result or {}).get("learning", {}).get("metrics")
    if metrics_payload:
        st.subheader("Outcome metrics")
        metrics = metrics_payload.get("metrics", [])
        if metrics:
            st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
            st.download_button("Download learning report", metrics_payload.get("report", ""), file_name="xauusd_learning_report.md")
        else:
            st.info("No completed outcomes yet. Let predictions expire, run another scan, or run backfill from a valid scan.")


def main() -> None:
    settings = load_yaml(ROOT / "config/default_settings.yaml")
    settings = apply_query_settings(settings)
    presets = load_yaml(ROOT / "config/presets.yaml")
    settings = settings_panel(settings, presets)
    refresh_panel()
    macro_bias = st.sidebar.selectbox("Macro bias", ["mixed", "supportive", "restrictive"], index=0)
    settings = data_panel(settings)

    st.title("XAU/USD Forecast Manager")
    st.caption("Advisory dashboard only. No broker execution. No auto-trading. Diagnostics first, signal second, no recommendation tuning.")
    page = st.sidebar.radio("Page", ["Forecast Manager", "KC Squeeze", "Market Map", "Macro / News", "Learning", "Settings", "Log Snapshot"])

    if page == "Forecast Manager":
        scan_label = "RESCAN NOW" if st.session_state.get("last_scan_result") else "SCAN NOW"
        if st.button(scan_label, type="primary", use_container_width=True):
            with st.spinner("Scanning XAU/USD and running strategy..."):
                scan_result = run_strategy_scan(settings, macro_bias)
                scan_result["learning"] = run_learning_cycle(scan_result)
                st.session_state["last_scan_result"] = scan_result

    result = st.session_state.get("last_scan_result")

    if page == "Forecast Manager":
        render_no_scan() if result is None else render_forecast_manager(result)
    elif page == "KC Squeeze":
        render_no_scan() if result is None else render_kc_page(result)
    elif page == "Market Map":
        render_no_scan() if result is None else render_market_map_page(result)
    elif page == "Macro / News":
        st.subheader("Macro context")
        st.json(macro_context(macro_bias, bool(settings.get("news_block_manual", False))) if result is None else result["macro"])
        st.write("Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches. API calendar can be added later.")
    elif page == "Learning":
        render_learning_page(result)
    elif page == "Settings":
        st.subheader("Active settings")
        st.json(safe_settings(settings))
        st.download_button("Download settings YAML", yaml.safe_dump(safe_settings(settings), sort_keys=False), file_name="active_settings.yaml")
    elif page == "Log Snapshot":
        render_no_scan() if result is None else render_log_snapshot(result)


main()
