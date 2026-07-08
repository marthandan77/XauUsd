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
from src.indicators import add_indicators
from src.kc_squeeze_engine import kc_squeeze_summary
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
    "sell_tp1_atr_multiplier",
    "sell_tp2_atr_multiplier",
    "sell_support_buffer_atr",
    "risk_per_trade_pct",
    "show_bollinger_bands",
    "show_keltner_channels",
    "news_block_manual",
]

st.set_page_config(page_title="XAU/USD Forecast Manager", layout="wide")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sample_freq(settings: dict) -> str:
    return {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1d"}.get(
        str(settings.get("price_interval", "15m")), "15min"
    )


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
    for key in ["last_scan_result", "last_scan_error", "last_scan_time"]:
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


def refresh_panel() -> None:
    st.sidebar.header("Refresh")
    if st.sidebar.button("Manual refresh page", type="secondary"):
        st.rerun()
    st.sidebar.caption("Manual page refresh does not rescan. Use SCAN NOW on Forecast Manager to recalculate.")


def fmt_price(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:,.2f}"


def fmt_units(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(value) or value <= 0:
        return "N/A"
    return f"{value:,.2f}"


def fmt_number(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except Exception:
        return "N/A"
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def has_plan_levels(plan: dict) -> bool:
    return plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None


def plan_value(plan: dict, key: str):
    if not has_plan_levels(plan):
        return None
    return plan.get(key)


def candidate_plan_from_scores(scores: dict, market, settings: dict) -> tuple[str, dict, str]:
    """Build non-executable candidate levels when the forecast is useful but official action is blocked/waiting."""
    if bool(settings.get("signals_disabled", False)):
        return "NO CANDIDATE", {"entry_zone_low": None, "entry_zone_high": None, "stop": None, "tp1": None, "tp2": None, "risk": 0.0, "note": "Signals disabled until real data is active."}, ""

    threshold = int(settings.get("forecast_threshold", 65))
    bias = str(scores.get("bias", "mixed"))
    bull_score = int(scores.get("bull_score", 0))
    bear_score = int(scores.get("bear_score", 0))

    if bool(scores.get("force_wait", False)):
        return "NO CANDIDATE", {"entry_zone_low": None, "entry_zone_high": None, "stop": None, "tp1": None, "tp2": None, "risk": 0.0, "note": scores.get("wait_reason", "Waiting for confirmation.")}, ""

    if bias == "bullish" and bull_score >= threshold and bool(settings.get("long_plans_enabled", True)):
        plan = build_trade_plan("BUY PLAN", market, settings)
        plan["note"] = "Bullish forecast candidate. Use only if final action is BUY PLAN and veto filters are clean."
        return "BUY CANDIDATE", plan, "Bullish candidate levels shown because bull forecast is above forecast threshold."

    if bias == "bearish" and bear_score >= threshold:
        preview_settings = dict(settings)
        preview_settings["short_plans_enabled"] = True
        plan = build_trade_plan("SELL PLAN", market, preview_settings)
        if bool(settings.get("short_plans_enabled", False)):
            plan["note"] = "Bearish forecast candidate. Use only if final action is SELL PLAN and veto filters are clean."
            return "SELL CANDIDATE", plan, "Bearish candidate levels shown because bear forecast is above forecast threshold."
        plan["note"] = "Bearish forecast preview only. Short plans are disabled, so this is not an active SELL PLAN."
        return "SELL PREVIEW ONLY", plan, "Bearish forecast met, but short plans are disabled in settings. Enable short plans for active SELL entry/TP/SL."

    return "NO CANDIDATE", {"entry_zone_low": None, "entry_zone_high": None, "stop": None, "tp1": None, "tp2": None, "risk": 0.0, "note": "No forecast candidate above threshold."}, ""


def plan_status(action: str, final_action: str, plan: dict, candidate_action: str, candidate_note: str) -> str:
    if final_action in ACTIONABLE:
        return f"Active {final_action}"
    if action in ACTIONABLE and has_plan_levels(plan):
        return f"Rejected {action}"
    if candidate_action != "NO CANDIDATE":
        return candidate_action
    return "No entry plan"


def settings_panel(settings: dict, presets: dict) -> dict:
    initialize_control_state(settings)

    left_col, right_col = st.sidebar.columns(2)
    reset_clicked = left_col.button("Reset", key="settings_reset_button", use_container_width=True)
    apply_clicked = right_col.button("Apply", key="settings_apply_button", type="primary", use_container_width=True)

    if reset_clicked:
        clear_query_settings()
        clear_control_state()
        st.rerun()

    st.sidebar.caption("Change controls first, then press Apply. Press SCAN NOW on Forecast Manager to recalculate.")
    st.sidebar.header("Forecast controls")

    if presets:
        selected = st.sidebar.selectbox("Preset", list(presets.keys()), index=0, key="preset_selector")
        if st.sidebar.button("Stage preset", key="stage_preset_button"):
            stage_preset_values(presets[selected])
            st.rerun()

    interval_options = ["5m", "15m", "30m", "1h", "4h", "1d"]
    if st.session_state.get(_control_key("price_interval")) not in interval_options:
        st.session_state[_control_key("price_interval")] = str(settings.get("price_interval", "15m"))
    period_options = ["7d", "30d", "60d", "6mo", "1y", "2y"]
    if st.session_state.get(_control_key("price_period")) not in period_options:
        st.session_state[_control_key("price_period")] = str(settings.get("price_period", "30d"))

    st.sidebar.selectbox("Timeframe", interval_options, key=_control_key("price_interval"))
    st.sidebar.selectbox("Lookback period", period_options, key=_control_key("price_period"))

    st.sidebar.slider("Bull threshold", 50, 95, key=_control_key("buy_threshold"))
    st.sidebar.slider("Bear threshold", 50, 95, key=_control_key("sell_threshold"))
    st.sidebar.slider("Watch threshold", 40, 90, key=_control_key("wait_threshold"))
    st.sidebar.slider("ATR guard multiplier", 0.8, 2.5, step=0.1, key=_control_key("atr_multiplier"))
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
    st.sidebar.slider("ATR stop multiplier", 1.0, 5.0, step=0.1, key=_control_key("atr_stop_multiplier"))
    st.sidebar.slider("ATR take-profit multiplier", 1.0, 10.0, step=0.1, key=_control_key("atr_tp_multiplier"))
    st.sidebar.slider("Sell Take Profit 1 ATR", 0.50, 3.00, step=0.05, key=_control_key("sell_tp1_atr_multiplier"))
    st.sidebar.slider("Sell Take Profit 2 ATR", 0.75, 5.00, step=0.05, key=_control_key("sell_tp2_atr_multiplier"))
    st.sidebar.slider("Sell support buffer ATR", 0.00, 1.00, step=0.05, key=_control_key("sell_support_buffer_atr"))
    st.sidebar.slider("Advisory risk %", 0.1, 2.0, step=0.1, key=_control_key("risk_per_trade_pct"))
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
            help="Use this only if Streamlit cannot see your Windows environment variable. The key is not saved to GitHub.",
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
    scan_settings["signals_disabled_reason"] = "Sample data source; KC trade signals are disabled until live API or uploaded real CSV is active."

    bars_raw = add_indicators(feed.bars, scan_settings)
    bars = bars_raw.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
    if bars.empty:
        return {
            "ok": False,
            "error": "Not enough clean OHLC data after indicator warm-up. Increase lookback period or use real CSV data.",
            "feed": feed,
            "settings": safe_settings(scan_settings),
            "scan_time": pd.Timestamp.now(tz="Asia/Singapore"),
        }

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
    veto = apply_veto(action, plan, market, regime, macro, scan_settings, latest_row)
    summary = explain(regime, scores, veto, market, macro)
    active_plan = veto["final_action"] in ACTIONABLE
    status = plan_status(action, veto["final_action"], plan, candidate_action, candidate_note)
    display_plan = plan if has_plan_levels(plan) else candidate_plan

    advisory_qty = 0.0
    if active_plan and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None:
        advisory_qty = advisory_position_size(
            account_equity=10000,
            risk_pct=float(scan_settings.get("risk_per_trade_pct", 0.5)) / 100.0,
            entry=(float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0,
            stop=float(plan["stop"]),
        )

    return {
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
    }


def price_chart(df: pd.DataFrame, settings: dict, action: str, plan: dict) -> go.Figure:
    chart_df = df.tail(220)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="XAU/USD",
        )
    )
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
        fig.add_trace(
            go.Scatter(
                x=[chart_df.index[-1]],
                y=[marker_price],
                mode="markers+text",
                text=[action],
                textposition="top center",
                name="Advisory signal",
            )
        )
    fig.update_layout(height=640, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render_no_scan() -> None:
    st.info("No scan result yet. Press SCAN NOW on Forecast Manager to fetch data, calculate indicators, and run the strategy.")


def render_source_metrics(result: dict) -> None:
    feed = result["feed"]
    bars = result.get("bars")
    market = result.get("market")
    source_cols = st.columns(5)
    source_cols[0].metric("Data source", feed.source)
    source_cols[1].metric("Latest price", fmt_price(market.price if market is not None else None))
    source_cols[2].metric("Rows loaded", f"{len(feed.bars):,}")
    source_cols[3].metric("Rows after warm-up", f"{len(bars):,}" if bars is not None else "0")
    source_cols[4].metric("Last scan", result["scan_time"].strftime("%H:%M:%S"))
    if feed.warning:
        st.warning(feed.warning)
    if bool(result["settings"].get("signals_disabled", False)):
        st.error(result["settings"].get("signals_disabled_reason", "Signals disabled."))


def render_forecast_manager(result: dict) -> None:
    render_source_metrics(result)
    if not result.get("ok"):
        st.error(result.get("error", "Scan failed."))
        return

    scores = result["scores"]
    veto = result["veto"]
    plan = result["plan"]
    display_plan = result["display_plan"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final action", veto["final_action"])
    c2.metric("Raw action", result["action"])
    c3.metric("Plan status", result["status"])
    c4.metric("Regime", result["regime"])
    c5.metric("Confidence", scores["confidence"])
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Bull score", scores["bull_score"])
    s2.metric("Bear score", scores["bear_score"])
    s3.metric("Range position", f"{result['market'].range_position_pct:.1f}%")
    s4.metric("Room ratio", f"{veto['room_ratio']:.2f}")
    st.info(result["summary"])
    if scores.get("force_wait") and scores.get("wait_reason"):
        st.warning(scores["wait_reason"])
    st.caption(plan.get("note", ""))
    if result["candidate_note"] and (not has_plan_levels(plan) or veto["final_action"] not in ACTIONABLE):
        st.warning(result["candidate_note"])
    if veto["reasons"]:
        st.error("Veto reasons: " + "; ".join(veto["reasons"]))
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Entry zone low", fmt_price(plan_value(display_plan, "entry_zone_low")))
    p2.metric("Entry zone high", fmt_price(plan_value(display_plan, "entry_zone_high")))
    p3.metric("Stop Loss", fmt_price(plan_value(display_plan, "stop")))
    p4.metric("Take Profit 1", fmt_price(plan_value(display_plan, "tp1")))
    p5, p6 = st.columns(2)
    p5.metric("Take Profit 2", fmt_price(plan_value(display_plan, "tp2")))
    p6.metric("Advisory units @ $10k equity", fmt_units(result["advisory_qty"]))
    if result["status"] != f"Active {veto['final_action']}":
        st.caption("Displayed levels are candidate/preview levels only. Execute manually only when final action is BUY PLAN or SELL PLAN and quality is Clean.")


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
        "close",
        "trend_sma",
        "bb_upper",
        "bb_lower",
        "kc_upper",
        "kc_lower",
        "compression_ratio",
        "squeeze_on",
        "squeeze_fired",
        "squeeze_recent",
        "bars_since_squeeze_release",
        "release_direction",
        "release_chase_atr",
        "release_bullish_confirmed",
        "release_bearish_confirmed",
        "kc_momentum",
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
        "regime": result["regime"],
        "kc_state": result["kc"]["state"],
        "kc_direction": result["kc"].get("release_direction"),
        "squeeze_on": result["kc"].get("squeeze_on"),
        "squeeze_fired": result["kc"].get("squeeze_fired"),
        "squeeze_recent": result["kc"].get("squeeze_recent"),
        "bars_since_squeeze_release": result["kc"].get("bars_since_squeeze_release"),
        "release_chase_atr": result["kc"].get("release_chase_atr"),
        "bias": result["scores"]["bias"],
        "confidence": result["scores"]["confidence"],
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
        "advisory_units_10k": result["advisory_qty"] if result["active_plan"] else None,
        "candidate_note": result["candidate_note"],
        "plan_note": result["plan"].get("note", ""),
    }
    snap = pd.DataFrame([row])
    st.dataframe(snap, use_container_width=True)
    st.download_button("Download snapshot CSV", snap.to_csv(index=False), file_name="xauusd_forecast_snapshot.csv")


settings = load_yaml(ROOT / "config/default_settings.yaml")
settings = apply_query_settings(settings)
presets = load_yaml(ROOT / "config/presets.yaml")
settings = settings_panel(settings, presets)
refresh_panel()
macro_bias = st.sidebar.selectbox("Macro bias", ["mixed", "supportive", "restrictive"], index=0)
settings = data_panel(settings)

st.title("XAU/USD Forecast Manager")
st.caption("Advisory dashboard only. No broker execution. No auto-trading. No backtesting. Veto first, signal second.")
page = st.sidebar.radio("Page", ["Forecast Manager", "KC Squeeze", "Market Map", "Macro / News", "Settings", "Log Snapshot"])

if page == "Forecast Manager":
    scan_label = "RESCAN NOW" if st.session_state.get("last_scan_result") else "SCAN NOW"
    if st.button(scan_label, type="primary", use_container_width=True):
        with st.spinner("Scanning XAU/USD and running strategy..."):
            st.session_state["last_scan_result"] = run_strategy_scan(settings, macro_bias)

result = st.session_state.get("last_scan_result")

if page == "Forecast Manager":
    if result is None:
        render_no_scan()
    else:
        render_forecast_manager(result)
elif page == "KC Squeeze":
    if result is None:
        render_no_scan()
    else:
        render_kc_page(result)
elif page == "Market Map":
    if result is None:
        render_no_scan()
    else:
        render_market_map_page(result)
elif page == "Macro / News":
    st.subheader("Macro context")
    if result is None:
        st.json(macro_context(macro_bias, bool(settings.get("news_block_manual", False))))
    else:
        st.json(result["macro"])
    st.write("Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches. API calendar can be added later.")
elif page == "Settings":
    st.subheader("Active settings")
    st.json(safe_settings(settings))
    st.download_button("Download settings YAML", yaml.safe_dump(safe_settings(settings), sort_keys=False), file_name="active_settings.yaml")
elif page == "Log Snapshot":
    if result is None:
        render_no_scan()
    else:
        render_log_snapshot(result)
