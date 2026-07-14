from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

st.set_page_config(page_title="XAU/USD Forecast Manager", layout="wide")


def undo_legacy_button_patch() -> None:
    """Undo the old Streamlit button monkey-patch if it is still alive in memory.

    Streamlit Cloud can rerun app.py without restarting the Python process. The previous
    learning controller patched DeltaGenerator.button, so the duplicate backfill button
    can survive even after the source file is fixed. This restores the original button
    method before any widgets are created.
    """
    try:
        from streamlit.delta_generator import DeltaGenerator

        current = DeltaGenerator.button
        original = getattr(current, "_xau_original_button", None)
        if original is not None:
            DeltaGenerator.button = original
    except Exception:
        pass


undo_legacy_button_patch()

ROOT = Path(__file__).resolve().parent


def _sample_freq(settings: dict) -> str:
    interval = str(settings.get("price_interval", "5m"))
    mapping = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
    return mapping.get(interval, "5min")


def _interval_minutes(settings: dict) -> float:
    return {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(
        str(settings.get("price_interval", "5m")), 5
    )


def validate_bars(bars: pd.DataFrame, settings: dict) -> str | None:
    """Returns an error string if bars are unusable, else None."""
    if bars.empty:
        return "No bars returned from data source."
    required = {"open", "high", "low", "close"}
    if not required.issubset(bars.columns):
        return f"Missing columns: {required - set(bars.columns)}"
    if bars[list(required)].isna().any().any():
        return "NaN values present in OHLC data."
    if (bars["high"] < bars["low"]).any():
        return "Invalid bars: high < low detected."
    if hasattr(bars.index, "tz") and bars.index.tz is not None:
        age_min = (pd.Timestamp.now(tz=bars.index.tz) - bars.index[-1]).total_seconds() / 60
        max_age = settings.get("stale_candle_multiplier", 3.0) * _interval_minutes(settings)
        if age_min > max_age:
            return f"Stale data: latest bar is {age_min:.0f} min old (limit {max_age:.0f} min)."
    return None
ACTIONABLE = {"BUY PLAN", "SELL PLAN"}
CONTROL_PREFIX = "control_"
PERSISTED_SETTING_KEYS = [
    "price_interval",
    "price_period",
    "buy_threshold",
    "sell_threshold",
    "wait_threshold",
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


@st.cache_resource
def engine() -> dict[str, Any]:
    from src.ai_explainer import explain
    from src.data_feed import FeedResult, load_csv, load_live_bars, make_sample_bars
    from src.forecast_engine import choose_action, score_forecast
    from src.gate_diagnostics import build_gate_breakdown, signal_stage, summarize_gate_blockers
    from src.horizon_forecast import build_multi_horizon_forecast
    from src.indicators import add_indicators
    from src.kc_squeeze_engine import kc_squeeze_summary
    from src.macro_engine import macro_context
    from src.market_map import build_market_map
    from src.range_model import price_bands
    from src.regime_engine import classify_regime
    from src.trade_plan import advisory_position_size, build_trade_plan
    from src.veto_engine import apply_veto
    return locals()


def safe_run_learning_cycle(result: dict) -> dict:
    try:
        from src.learning.controller import run_learning_cycle

        return run_learning_cycle(result)
    except Exception as exc:
        return {"ok": False, "reason": f"Learning cycle skipped: {exc}"}


def safe_refresh_learning_metrics() -> dict:
    try:
        from src.learning.controller import refresh_learning_metrics

        return refresh_learning_metrics()
    except Exception as exc:
        return {"ok": False, "reason": f"Learning metrics unavailable: {exc}", "metrics": [], "report": ""}


def safe_backfill_learning_history(result: dict | None) -> dict:
    try:
        from src.learning.controller import backfill_learning_history

        return backfill_learning_history(result or {})
    except Exception as exc:
        return {"ok": False, "reason": f"Backfill unavailable: {exc}"}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def safe_settings(settings: dict) -> dict:
    return {key: value for key, value in settings.items() if key != "uploaded_file"}


def _control_key(key: str) -> str:
    return f"{CONTROL_PREFIX}{key}"


def _query_params_as_dict() -> dict[str, str]:
    try:
        return {key: value for key, value in st.query_params.items()}
    except Exception:
        return {}


def _coerce_query_value(raw_value, current_value):
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


def persist_query_settings(settings: dict) -> None:
    try:
        st.query_params.clear()
        for key in PERSISTED_SETTING_KEYS:
            if key not in settings:
                continue
            value = settings[key]
            st.query_params[key] = "true" if value is True else "false" if value is False else str(value)
    except Exception:
        pass


def clear_query_settings() -> None:
    try:
        st.query_params.clear()
    except Exception:
        pass


def initialize_control_state(settings: dict) -> None:
    for key in PERSISTED_SETTING_KEYS:
        if key in settings and _control_key(key) not in st.session_state:
            st.session_state[_control_key(key)] = settings[key]


def clear_control_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(CONTROL_PREFIX) or key in {"last_scan_result", "learning_metrics", "learning_backfill"}:
            st.session_state.pop(key, None)


def staged_settings(settings: dict) -> dict:
    staged = dict(settings)
    for key in PERSISTED_SETTING_KEYS:
        control_key = _control_key(key)
        if control_key in st.session_state:
            staged[key] = st.session_state[control_key]
    staged["long_plans_enabled"] = True
    staged["short_plans_enabled"] = True
    return staged


def fmt_price(value) -> str:
    try:
        number = float(value)
    except Exception:
        return "N/A"
    return f"{number:,.2f}" if math.isfinite(number) else "N/A"


def fmt_number(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return "N/A"
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def plan_value(plan: dict, key: str):
    return plan.get(key) if plan and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None else None


def settings_panel(settings: dict, presets: dict) -> dict:
    initialize_control_state(settings)
    reset_col, apply_col = st.sidebar.columns(2)
    reset_clicked = reset_col.button("Reset", key="settings_reset_button", use_container_width=True)
    apply_clicked = apply_col.button("Apply", key="settings_apply_button", type="primary", use_container_width=True)

    result = st.session_state.get("last_scan_result")
    backfill_ready = isinstance(result, dict) and bool(result.get("ok")) and result.get("bars") is not None
    backfill_clicked = st.sidebar.button(
        "Backfill Learning History",
        key="learning_backfill_sidebar_direct_v3",
        disabled=not backfill_ready,
        use_container_width=True,
        help="Run SCAN NOW first. Backfill uses the latest scanned bars to create historical walk-forward outcome samples.",
    )
    st.sidebar.caption("Backfill is enabled only after SCAN NOW. It is for diagnosis, not auto-optimization.")

    if reset_clicked:
        clear_query_settings()
        clear_control_state()
        st.rerun()

    if backfill_clicked:
        with st.sidebar:
            with st.spinner("Backfilling learning history..."):
                st.session_state["learning_backfill"] = safe_backfill_learning_history(result)
                st.session_state["learning_metrics"] = safe_refresh_learning_metrics()
        status = st.session_state["learning_backfill"]
        if status.get("ok"):
            st.sidebar.success(f"Backfilled {status.get('outcome_count', 0)} outcomes.")
        else:
            st.sidebar.warning(status.get("reason", "Backfill failed."))

    st.sidebar.caption("Change controls first, press Apply, then press SCAN NOW to recalculate.")
    st.sidebar.header("Forecast controls")

    interval_options = ["5m", "15m", "30m", "1h", "4h", "1d"]
    period_options = ["7d", "14d", "30d", "60d", "6mo", "1y", "2y"]
    if st.session_state.get(_control_key("price_interval")) not in interval_options:
        st.session_state[_control_key("price_interval")] = str(settings.get("price_interval", "5m"))
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
    st.sidebar.toggle("Show Bollinger Bands", key=_control_key("show_bollinger_bands"))
    st.sidebar.toggle("Show Keltner Channels", key=_control_key("show_keltner_channels"))
    st.sidebar.toggle("Manual event block", key=_control_key("news_block_manual"))

    if apply_clicked:
        applied = staged_settings(settings)
        persist_query_settings(applied)
        st.sidebar.success("Settings applied. Press SCAN NOW to recalculate.")
        return applied

    return staged_settings(settings)


def data_panel(settings: dict) -> dict:
    e = engine()
    st.sidebar.header("Data")
    data_mode = st.sidebar.radio("Data mode", ["Twelve Data API", "CSV upload", "Sample"], index=0, key="data_mode")
    settings["data_mode"] = data_mode
    settings["uploaded_file"] = None
    if data_mode == "CSV upload":
        settings["uploaded_file"] = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
    elif data_mode == "Twelve Data API":
        key_input = st.sidebar.text_input("Twelve Data API key - optional local session override", value="", type="password", key="twelve_data_runtime_key")
        if key_input.strip():
            settings["twelve_data_api_key_runtime"] = key_input.strip()
    return settings


def fetch_bars(settings: dict):
    e = engine()
    data_mode = str(settings.get("data_mode", "Twelve Data API"))
    if data_mode == "CSV upload":
        uploaded = settings.get("uploaded_file")
        if uploaded is not None:
            return e["load_csv"](uploaded)
        return e["FeedResult"](e["make_sample_bars"](freq=_sample_freq(settings)), "sample", "CSV not uploaded; sample data used")
    if data_mode == "Twelve Data API":
        feed = e["load_live_bars"](settings)
        if not feed.bars.empty:
            return feed
        return e["FeedResult"](e["make_sample_bars"](freq=_sample_freq(settings)), "sample", "Twelve Data unavailable; sample data used. " + feed.warning)
    return e["FeedResult"](e["make_sample_bars"](freq=_sample_freq(settings)), "sample", "sample data")


def run_strategy_scan(settings: dict, macro_bias: str) -> dict:
    e = engine()
    scan_settings = dict(settings)
    try:
        feed = fetch_bars(scan_settings)
        error = validate_bars(feed.bars, scan_settings)
        if error:
            return {"ok": False, "error": error, "feed": feed, "settings": safe_settings(scan_settings),
                    "scan_time": pd.Timestamp.now(tz="Asia/Singapore"), "multi_horizon": []}

        scan_settings["signals_disabled"] = feed.source == "sample"
        scan_settings["signals_disabled_reason"] = "Sample data source; trade signals are disabled until live API or uploaded real CSV is active."

        bars_raw = e["add_indicators"](feed.bars, scan_settings)
        bars = bars_raw.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
        if bars.empty:
            return {"ok": False, "error": "Not enough clean OHLC data after indicator warm-up.",
                    "feed": feed, "settings": safe_settings(scan_settings),
                    "scan_time": pd.Timestamp.now(tz="Asia/Singapore"), "multi_horizon": []}

        min_needed = max(scan_settings.get("horizon_model_lookback_bars", 500) // 5, 100)
        if len(bars) < min_needed:
            return {"ok": False, "error": f"Only {len(bars)} clean bars; need at least {min_needed}.",
                    "feed": feed, "settings": safe_settings(scan_settings),
                    "scan_time": pd.Timestamp.now(tz="Asia/Singapore"), "multi_horizon": []}

        multi_horizon = e["build_multi_horizon_forecast"](bars, scan_settings)
        kc = e["kc_squeeze_summary"](bars, scan_settings)
        latest_row = bars.iloc[-1].to_dict()
        latest_row["kc_state"] = kc["state"]
        latest_row["kc_reason"] = kc["reason"]

        market = e["build_market_map"](bars, scan_settings)
        regime = e["classify_regime"](bars, market, scan_settings)
        macro = e["macro_context"](macro_bias, bool(scan_settings.get("news_block_manual", False)))
        scores = e["score_forecast"](latest_row, market, regime, macro, scan_settings)
        if kc["state"] != "insufficient_data":
            scores["kc_state"] = kc["state"]
            scores["kc_reason"] = kc["reason"]

        action = e["choose_action"](scores, scan_settings)
        plan = e["build_trade_plan"](action, market, scan_settings)
        bands = e["price_bands"](market, regime)
        veto = e["apply_veto"](action, plan, market, regime, macro, scan_settings, latest_row, multi_horizon)
        summary = e["explain"](regime, scores, veto, market, macro)

        active_plan = veto["final_action"] in ACTIONABLE
        advisory_qty = 0.0
        if active_plan and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None:
            account_size = float(scan_settings.get("account_size", 10000))
            advisory_qty = e["advisory_position_size"](
                account_size,
                float(scan_settings.get("risk_per_trade_pct", 0.5)) / 100.0,
                (float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0,
                float(plan["stop"]),
            )

        result = {
            "ok": True, "scan_time": pd.Timestamp.now(tz="Asia/Singapore"), "feed": feed,
            "settings": safe_settings(scan_settings), "bars": bars, "kc": kc, "latest_row": latest_row,
            "market": market, "regime": regime, "macro": macro, "scores": scores, "action": action,
            "plan": plan, "display_plan": plan, "bands": bands, "veto": veto, "summary": summary,
            "active_plan": active_plan, "status": "Allowed" if active_plan else "No active trade",
            "advisory_qty": advisory_qty, "multi_horizon": multi_horizon, "candidate_note": "",
        }
        result["signal_stage"] = e["signal_stage"](result)
        result["gate_breakdown"] = e["build_gate_breakdown"](result)
        result["gate_summary"] = e["summarize_gate_blockers"](result)
        return result

    except Exception as exc:
        import traceback
        return {
            "ok": False,
            "error": f"Scan failed: {exc}",
            "traceback": traceback.format_exc(),
            "settings": safe_settings(scan_settings),
            "scan_time": pd.Timestamp.now(tz="Asia/Singapore"),
            "multi_horizon": [],
        }


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
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render_no_scan() -> None:
    st.info("No scan result yet. Press SCAN NOW on Forecast Manager to fetch data, calculate indicators, and run the strategy.")


def horizon_rows(forecasts: list[dict]) -> list[dict]:
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


def render_forecast_manager(result: dict) -> None:
    if not result.get("ok"):
        st.error(result.get("error", "Scan failed."))
        return
    feed = result["feed"]
    market = result["market"]
    scores = result["scores"]
    veto = result["veto"]
    cols = st.columns(5)
    cols[0].metric("Data source", feed.source)
    cols[1].metric("Latest price", fmt_price(market.price))
    cols[2].metric("Rows after warm-up", f"{len(result['bars']):,}")
    cols[3].metric("Signal stage", result.get("signal_stage", "N/A"))
    cols[4].metric("Final action", veto.get("final_action"))
    if feed.warning:
        st.warning(feed.warning)
    if bool(result["settings"].get("signals_disabled", False)):
        st.error(result["settings"].get("signals_disabled_reason", "Signals disabled."))

    st.subheader("Trade permission")
    if veto.get("final_action") in ACTIONABLE:
        st.success(f"{veto.get('final_action')} — active advisory plan allowed")
    else:
        st.warning(f"{veto.get('final_action')} — no active trade")
    c1, c2, c3 = st.columns(3)
    c1.metric("Buyer strength", f"{int(scores.get('bull_score', 0))}/100")
    c2.metric("Seller strength", f"{int(scores.get('bear_score', 0))}/100")
    c3.metric("Room ratio", fmt_number(veto.get("room_ratio"), 2))

    st.subheader("Gate Breakdown")
    st.dataframe(pd.DataFrame(result.get("gate_breakdown", [])), use_container_width=True, hide_index=True)
    st.caption("This table shows which module is blocking, watching, or passing.")

    st.subheader("5m Entry + Higher-Timeframe Filters")
    st.dataframe(pd.DataFrame(horizon_rows(result.get("multi_horizon", []))), use_container_width=True, hide_index=True)

    st.subheader("Trade levels")
    plan = result.get("plan", {})
    if veto.get("final_action") in ACTIONABLE:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry low", fmt_price(plan_value(plan, "entry_zone_low")))
        p2.metric("Entry high", fmt_price(plan_value(plan, "entry_zone_high")))
        p3.metric("Stop", fmt_price(plan_value(plan, "stop")))
        p4.metric("TP1", fmt_price(plan_value(plan, "tp1")))
    else:
        st.write("No active trade plan. Levels hidden to avoid accidental trading.")
    with st.expander("Advanced details"):
        st.write(result.get("summary"))
        st.write(result.get("gate_summary"))
        st.json({"regime": result.get("regime"), "kc": result.get("kc"), "veto": result.get("veto")})
    st.plotly_chart(price_chart(result["bars"], result["settings"], veto.get("final_action"), plan), use_container_width=True)


def render_learning_page(result: dict | None) -> None:
    st.subheader("GitHub Learning")
    st.write("Learning logs predictions and reads outcomes/backfills. It does not generate recommended settings.")
    left, right = st.columns(2)
    if left.button("Refresh learning metrics", use_container_width=True):
        st.session_state["learning_metrics"] = safe_refresh_learning_metrics()
    if right.button("Backfill from last scan", use_container_width=True, disabled=result is None or not result.get("ok", False)):
        st.session_state["learning_backfill"] = safe_backfill_learning_history(result)
        st.session_state["learning_metrics"] = safe_refresh_learning_metrics()
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
            st.info(metrics_payload.get("reason", "No completed outcomes yet."))


def render_log_snapshot(result: dict) -> None:
    if not result or not result.get("ok"):
        render_no_scan()
        return
    snap = pd.DataFrame([{
        "scan_time": result["scan_time"].isoformat(),
        "source": result["feed"].source,
        "price": result["market"].price,
        "signal_stage": result.get("signal_stage"),
        "gate_summary": result.get("gate_summary"),
        "raw_action": result.get("action"),
        "final_action": result["veto"].get("final_action"),
        "veto_reasons": "; ".join(result["veto"].get("reasons", [])),
    }])
    gates = pd.DataFrame(result.get("gate_breakdown", []))
    horizon = pd.DataFrame(horizon_rows(result.get("multi_horizon", [])))
    st.subheader("Scan snapshot")
    st.dataframe(snap, use_container_width=True)
    st.download_button("Download snapshot CSV", snap.to_csv(index=False), file_name="xauusd_forecast_snapshot.csv")
    st.subheader("Gate snapshot")
    st.dataframe(gates, use_container_width=True, hide_index=True)
    st.download_button("Download gate CSV", gates.to_csv(index=False), file_name="xauusd_gate_snapshot.csv")
    st.subheader("Horizon snapshot")
    st.dataframe(horizon, use_container_width=True, hide_index=True)
    st.download_button("Download horizon CSV", horizon.to_csv(index=False), file_name="xauusd_horizon_snapshot.csv")


def main() -> None:
    try:
        settings = load_yaml(ROOT / "config/default_settings.yaml")
        settings = apply_query_settings(settings)
        presets = load_yaml(ROOT / "config/presets.yaml")
        settings = settings_panel(settings, presets)
        st.sidebar.header("Refresh")
        if st.sidebar.button("Manual refresh page", type="secondary"):
            st.rerun()
        macro_bias = st.sidebar.selectbox("Macro bias", ["mixed", "supportive", "restrictive"], index=0)
        settings = data_panel(settings)
    except Exception as exc:
        st.error(f"Sidebar failed to load: {exc}")
        st.stop()

    st.title("XAU/USD Forecast Manager")
    st.caption("Advisory dashboard only. No broker execution. Diagnostics first, signal second, no recommendation tuning.")
    page = st.sidebar.radio("Page", ["Forecast Manager", "KC Squeeze", "Market Map", "Macro / News", "Learning", "Settings", "Log Snapshot"])

    if page == "Forecast Manager":
        scan_label = "RESCAN NOW" if st.session_state.get("last_scan_result") else "SCAN NOW"
        if st.button(scan_label, type="primary", use_container_width=True):
            with st.spinner("Scanning XAU/USD and running strategy..."):
                scan_result = run_strategy_scan(settings, macro_bias)
                scan_result["learning"] = safe_run_learning_cycle(scan_result)
                st.session_state["last_scan_result"] = scan_result

    result = st.session_state.get("last_scan_result")
    if page == "Forecast Manager":
        render_no_scan() if result is None else render_forecast_manager(result)
    elif page == "KC Squeeze":
        render_no_scan() if result is None else st.json(result.get("kc", {}))
    elif page == "Market Map":
        render_no_scan() if result is None else st.json(result["market"].to_dict())
    elif page == "Macro / News":
        st.subheader("Macro context")
        st.json(engine()["macro_context"](macro_bias, bool(settings.get("news_block_manual", False))) if result is None else result["macro"])
        st.write("Use manual event block around CPI, NFP, PCE, FOMC and major Fed speeches.")
    elif page == "Learning":
        render_learning_page(result)
    elif page == "Settings":
        st.subheader("Active settings")
        st.json(safe_settings(settings))
        st.download_button("Download settings YAML", yaml.safe_dump(safe_settings(settings), sort_keys=False), file_name="active_settings.yaml")
    elif page == "Log Snapshot":
        render_log_snapshot(result)


main()
