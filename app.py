from __future__ import annotations

import base64
import json
import math
from io import BytesIO
from pathlib import Path

import numpy as np
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
from src.quant_forecast import build_unified_forecast, scan_validity
from src.range_model import price_bands
from src.regime_engine import classify_regime
from src.trade_plan import advisory_position_size, build_trade_plan
from src.veto_engine import apply_veto

ROOT = Path(__file__).resolve().parent
ACTIONABLE = {"BUY PLAN", "SELL PLAN"}
DATA_MODES = ["Twelve Data", "CSV upload", "Sample"]
MACRO_BIASES = ["mixed", "supportive", "restrictive"]
INTERVAL_OPTIONS = ["15m", "30m", "1h", "4h", "1d"]
PERIOD_OPTIONS = ["7d", "30d", "60d", "6mo", "1y", "2y"]
DRAFT_SETTING_KEYS = [
    "twelve_data_symbol",
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
    "trend_length",
    "atr_period",
    "atr_stop_multiplier",
    "atr_tp_multiplier",
    "risk_per_trade_pct",
    "long_plans_enabled",
    "short_plans_enabled",
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
    return {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}.get(
        str(settings.get("price_interval", "15m")), "15min"
    )


def _sample_fallback(settings: dict, warning: str) -> FeedResult:
    if not bool(settings.get("sample_data_enabled", True)):
        return FeedResult(pd.DataFrame(), "none", warning)
    return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", warning)


def fmt_price(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{value:,.2f}" if math.isfinite(value) else "N/A"


def fmt_units(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{value:,.2f}" if math.isfinite(value) and value > 0 else "N/A"


def fmt_percent(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{100.0 * value:.1f}%" if math.isfinite(value) else "N/A"


def fmt_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def active_value(plan: dict, key: str, final_action: str):
    return plan.get(key) if final_action in ACTIONABLE else None


def _draft_key(setting_key: str) -> str:
    return f"draft_{setting_key}"


def _encode_persisted_config(settings: dict, data_mode: str, macro_bias: str) -> str:
    payload = {"settings": settings, "data_mode": data_mode, "macro_bias": macro_bias}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _read_persisted_config() -> dict:
    token = st.query_params.get("config")
    if not token:
        return {}
    if isinstance(token, list):
        token = token[-1]
    try:
        padded = str(token) + "=" * (-len(str(token)) % 4)
        value = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        payload = json.loads(value)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_persisted_config(settings: dict, data_mode: str, macro_bias: str) -> None:
    st.query_params["config"] = _encode_persisted_config(settings, data_mode, macro_bias)


def _clear_app_state() -> None:
    st.query_params.clear()
    managed = {
        "applied_settings",
        "applied_data_mode",
        "applied_macro_bias",
        "applied_csv_bytes",
        "applied_csv_name",
        "scan_result",
        "draft_state_initialized",
        "selected_preset",
        "sidebar_notice",
        "draft_uploaded_csv",
    }
    for key in list(st.session_state.keys()):
        if key in managed or key.startswith("draft_"):
            del st.session_state[key]


def _sync_draft_state(settings: dict) -> None:
    for setting_key in DRAFT_SETTING_KEYS:
        st.session_state[_draft_key(setting_key)] = settings.get(setting_key)


def _initialize_session(default_settings: dict) -> None:
    persisted = _read_persisted_config()
    persisted_settings = persisted.get("settings") if isinstance(persisted.get("settings"), dict) else {}
    if "applied_settings" not in st.session_state:
        applied = dict(default_settings)
        for key, value in persisted_settings.items():
            if key in applied:
                applied[key] = value
        st.session_state.applied_settings = applied
    if "applied_data_mode" not in st.session_state:
        mode = str(persisted.get("data_mode", "Twelve Data"))
        st.session_state.applied_data_mode = mode if mode in DATA_MODES else "Twelve Data"
    if "applied_macro_bias" not in st.session_state:
        bias = str(persisted.get("macro_bias", "mixed"))
        st.session_state.applied_macro_bias = bias if bias in MACRO_BIASES else "mixed"
    st.session_state.setdefault("applied_csv_bytes", None)
    st.session_state.setdefault("applied_csv_name", None)
    st.session_state.setdefault("scan_result", None)
    if "draft_state_initialized" not in st.session_state:
        _sync_draft_state(st.session_state.applied_settings)
        st.session_state.draft_data_mode = st.session_state.applied_data_mode
        st.session_state.draft_macro_bias = st.session_state.applied_macro_bias
        st.session_state.draft_state_initialized = True


def _load_selected_preset(default_settings: dict, presets: dict, selected_name: str) -> None:
    preset_settings = dict(default_settings)
    preset_settings.update(presets.get(selected_name, {}))
    _sync_draft_state(preset_settings)
    st.session_state.sidebar_notice = f"{selected_name} loaded as draft. Press Apply Settings to activate it."


def settings_panel(default_settings: dict, presets: dict) -> None:
    st.sidebar.header("Forecast controls")
    if presets:
        selected = st.sidebar.selectbox("Preset", list(presets.keys()), key="selected_preset")
        if st.sidebar.button("Load preset", use_container_width=True):
            _load_selected_preset(default_settings, presets, selected)
            st.rerun()

    notice = st.session_state.pop("sidebar_notice", None)
    if notice:
        st.sidebar.success(notice)

    with st.sidebar.form("settings_form", clear_on_submit=False):
        reset_col, apply_col = st.columns(2)
        reset_clicked = reset_col.form_submit_button("Reset", use_container_width=True)
        apply_clicked = apply_col.form_submit_button("Apply Settings", type="primary", use_container_width=True)

        st.subheader("Data")
        data_mode = st.radio("Data mode", DATA_MODES, key="draft_data_mode")
        symbol = st.text_input("Twelve Data symbol", key="draft_twelve_data_symbol", disabled=data_mode != "Twelve Data")
        uploaded = st.file_uploader(
            "Upload OHLC CSV", type=["csv"], key="draft_uploaded_csv", disabled=data_mode != "CSV upload"
        )
        macro_bias = st.selectbox("Macro bias", MACRO_BIASES, key="draft_macro_bias")

        st.subheader("Forecast")
        price_interval = st.selectbox("Timeframe", INTERVAL_OPTIONS, key="draft_price_interval")
        price_period = st.selectbox("Lookback period", PERIOD_OPTIONS, key="draft_price_period")
        buy_threshold = st.slider("Bull threshold", 50, 95, key="draft_buy_threshold")
        sell_threshold = st.slider("Bear threshold", 50, 95, key="draft_sell_threshold")
        wait_threshold = st.slider("Watch threshold", 40, 90, key="draft_wait_threshold")
        atr_multiplier = st.slider("ATR guard multiplier", 0.8, 2.5, step=0.1, key="draft_atr_multiplier")
        min_reward_risk = st.slider("Minimum room ratio", 1.0, 3.0, step=0.1, key="draft_min_reward_risk")
        middle_range_lower_pct = st.slider("Middle zone lower %", 10, 49, key="draft_middle_range_lower_pct")
        middle_range_upper_pct = st.slider("Middle zone upper %", 51, 90, key="draft_middle_range_upper_pct")

        st.subheader("KC Squeeze and risk")
        kc_squeeze_enabled = st.toggle("Enable KC Squeeze module", key="draft_kc_squeeze_enabled")
        bb_length = st.slider("BB length", 10, 60, key="draft_bb_length")
        bb_mult = st.slider("BB multiplier", 1.0, 3.5, step=0.1, key="draft_bb_mult")
        kc_length = st.slider("KC length", 10, 60, key="draft_kc_length")
        kc_mult = st.slider("KC multiplier", 1.0, 3.5, step=0.1, key="draft_kc_mult")
        trend_length = st.slider("Trend SMA length", 50, 300, key="draft_trend_length")
        atr_period = st.slider("ATR period", 5, 50, key="draft_atr_period")
        atr_stop_multiplier = st.slider("ATR stop multiplier", 1.0, 5.0, step=0.1, key="draft_atr_stop_multiplier")
        atr_tp_multiplier = st.slider("ATR take-profit multiplier", 1.0, 10.0, step=0.1, key="draft_atr_tp_multiplier")
        risk_per_trade_pct = st.slider("Advisory risk %", 0.1, 2.0, step=0.1, key="draft_risk_per_trade_pct")
        long_plans_enabled = st.toggle("Enable long plans", key="draft_long_plans_enabled")
        short_plans_enabled = st.toggle("Enable short plans", key="draft_short_plans_enabled")
        show_bollinger_bands = st.toggle("Show Bollinger Bands", key="draft_show_bollinger_bands")
        show_keltner_channels = st.toggle("Show Keltner Channels", key="draft_show_keltner_channels")
        news_block_manual = st.toggle("Manual event block", key="draft_news_block_manual")

    if reset_clicked:
        _clear_app_state()
        st.rerun()

    if apply_clicked:
        updated = dict(st.session_state.applied_settings)
        updated.update(
            {
                "twelve_data_symbol": symbol.strip() or "XAU/USD",
                "price_interval": price_interval,
                "price_period": price_period,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "wait_threshold": wait_threshold,
                "atr_multiplier": atr_multiplier,
                "min_reward_risk": min_reward_risk,
                "middle_range_lower_pct": middle_range_lower_pct,
                "middle_range_upper_pct": middle_range_upper_pct,
                "kc_squeeze_enabled": kc_squeeze_enabled,
                "bb_length": bb_length,
                "bb_mult": bb_mult,
                "kc_length": kc_length,
                "kc_mult": kc_mult,
                "trend_length": trend_length,
                "atr_period": atr_period,
                "atr_stop_multiplier": atr_stop_multiplier,
                "atr_tp_multiplier": atr_tp_multiplier,
                "risk_per_trade_pct": risk_per_trade_pct,
                "long_plans_enabled": long_plans_enabled,
                "short_plans_enabled": short_plans_enabled,
                "show_bollinger_bands": show_bollinger_bands,
                "show_keltner_channels": show_keltner_channels,
                "news_block_manual": news_block_manual,
            }
        )
        st.session_state.applied_settings = updated
        st.session_state.applied_data_mode = data_mode
        st.session_state.applied_macro_bias = macro_bias
        if uploaded is not None:
            st.session_state.applied_csv_bytes = uploaded.getvalue()
            st.session_state.applied_csv_name = uploaded.name
        _write_persisted_config(updated, data_mode, macro_bias)
        st.session_state.scan_result = None
        st.session_state.sidebar_notice = "Settings applied. Press Scan Market to run a fresh analysis."
        st.rerun()

    st.sidebar.caption(
        "Applied settings survive browser refreshes. Draft changes do not affect a scan until Apply Settings is pressed."
    )


def load_bars(settings: dict, data_mode: str, csv_bytes: bytes | None) -> FeedResult:
    if data_mode == "CSV upload":
        if not csv_bytes:
            return _sample_fallback(settings, "CSV not uploaded; sample data used")
        feed = load_csv(BytesIO(csv_bytes))
        return feed if not feed.bars.empty else _sample_fallback(settings, f"{feed.warning}; sample data used")
    if data_mode == "Twelve Data":
        feed = load_live_bars(settings)
        return feed if not feed.bars.empty else _sample_fallback(settings, f"{feed.warning}; sample data used")
    return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "sample data selected")


def run_scan(settings: dict, macro_bias: str, data_mode: str, csv_bytes: bytes | None) -> dict:
    scanned_at = pd.Timestamp.now(tz="UTC")
    feed = load_bars(settings, data_mode, csv_bytes)
    if feed.bars.empty:
        return {"error": "No market data is available. Check the data source, API secret, or CSV file.", "feed": feed}

    bars_raw = add_indicators(feed.bars, settings)
    bars = bars_raw.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
    if bars.empty:
        return {
            "error": "Not enough clean OHLC data after indicator warm-up. Increase the lookback or enable sample data.",
            "feed": feed,
        }

    kc = kc_squeeze_summary(bars, settings)
    latest_row = bars.iloc[-1].to_dict()
    latest_row["kc_state"] = kc["state"]
    latest_row["kc_reason"] = kc["reason"]
    market = build_market_map(bars, settings)
    regime = classify_regime(bars, market, settings)
    macro = macro_context(macro_bias, bool(settings.get("news_block_manual", False)))
    scores = score_forecast(latest_row, market, regime, macro, settings)
    if kc["state"] != "insufficient_data":
        scores["kc_state"] = kc["state"]
        scores["kc_reason"] = kc["reason"]

    action = choose_action(scores, settings)
    plan = build_trade_plan(action, market, settings)
    bands = price_bands(market, regime)
    rule_veto = apply_veto(action, plan, market, regime, macro, settings)
    quant = build_unified_forecast(bars, scores, plan, rule_veto, settings, scanned_at)
    if quant.get("ready"):
        quant["scan_regime"] = regime
        quant["scan_atr"] = float(market.atr)

    veto = dict(rule_veto)
    veto["rule_action"] = rule_veto["final_action"]
    if quant.get("ready"):
        veto["final_action"] = quant["final_action"]
        quant_reasons = [f"quant: {reason}" for reason in quant.get("quant_reasons", [])]
        veto["reasons"] = list(veto.get("reasons", [])) + quant_reasons
        if quant_reasons:
            veto["trade_quality"] = "Rejected by quant"

    summary = explain(regime, scores, veto, market, macro)
    if quant.get("ready"):
        summary += (
            f" Quant median {quant['expected_price']:.2f} over {quant['horizon_bars']} bars; "
            f"up probability {100 * quant['probability_up']:.1f}%; mode {quant['mode']}."
        )

    active_plan = veto["final_action"] in ACTIONABLE
    advisory_qty = 0.0
    if active_plan and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None and plan.get("stop") is not None:
        advisory_qty = advisory_position_size(
            account_equity=10000,
            risk_pct=float(settings.get("risk_per_trade_pct", 0.5)) / 100.0,
            entry=(float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0,
            stop=float(plan["stop"]),
        )

    return {
        "error": "",
        "scanned_at": scanned_at,
        "settings": dict(settings),
        "macro_bias": macro_bias,
        "data_mode": data_mode,
        "feed": feed,
        "bars": bars,
        "kc": kc,
        "market": market,
        "regime": regime,
        "macro": macro,
        "scores": scores,
        "action": action,
        "plan": plan,
        "bands": bands,
        "rule_veto": rule_veto,
        "veto": veto,
        "summary": summary,
        "quant": quant,
        "active_plan": active_plan,
        "advisory_qty": advisory_qty,
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_live_validity_snapshot(settings_json: str) -> dict:
    settings = json.loads(settings_json)
    status_settings = dict(settings)
    minimum = max(int(status_settings.get("trend_length", 200)) + 50, 300)
    status_settings["minimum_bars"] = minimum
    status_settings["twelve_data_outputsize"] = max(
        minimum,
        min(int(status_settings.get("twelve_data_outputsize", 5000)), 1000),
    )
    feed = load_live_bars(status_settings)
    if feed.bars.empty:
        return {"warning": feed.warning}
    enriched = add_indicators(feed.bars, status_settings)
    clean = enriched.dropna(subset=["close", "atr", "ema_fast", "ema_slow", "trend_sma"]).copy()
    if clean.empty:
        return {"warning": "Live validity bars did not complete indicator warm-up."}
    market = build_market_map(clean, status_settings)
    regime = classify_regime(clean, market, status_settings)
    return {
        "price": float(market.price),
        "atr": float(market.atr),
        "regime": regime,
        "refreshed_at": pd.Timestamp.now(tz="UTC"),
        "warning": feed.warning,
    }


def _effective_veto(result: dict, validity: dict) -> dict:
    veto = dict(result["veto"])
    veto["reasons"] = list(veto.get("reasons", []))
    if validity.get("status") == "EXPIRED" and veto.get("final_action") in ACTIONABLE:
        veto["final_action"] = "HOLD"
        veto["trade_quality"] = "Expired"
        veto["reasons"].append("scan expired; rescan required")
    return veto


def price_chart(df: pd.DataFrame, settings: dict, action: str, plan: dict, quant: dict | None = None) -> go.Figure:
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

    if quant and quant.get("ready"):
        horizon = int(quant["horizon_bars"])
        q = np.linspace(0.0, 1.0, horizon + 1)
        path_start = pd.Timestamp(chart_df.index[-1])
        path_end = pd.Timestamp(quant["expiry_time"])
        if path_start.tzinfo is None and path_end.tzinfo is not None:
            path_end = path_end.tz_localize(None)
        elif path_start.tzinfo is not None and path_end.tzinfo is None:
            path_end = path_end.tz_localize(path_start.tzinfo)
        elif path_start.tzinfo is not None and path_end.tzinfo is not None:
            path_end = path_end.tz_convert(path_start.tzinfo)
        path_index = pd.date_range(start=path_start, end=path_end, periods=horizon + 1)
        anchor = float(quant["anchor_price"])
        mu = float(quant["expected_log_return"])
        half_width = float(quant["selected_diagnostics"]["conformal_half_width"])
        expected = anchor * np.exp(mu * q)
        lower = anchor * np.exp(mu * q - half_width * np.sqrt(q))
        upper = anchor * np.exp(mu * q + half_width * np.sqrt(q))
        fig.add_trace(go.Scatter(x=path_index, y=expected, mode="lines+markers", name="Quant expected path"))
        fig.add_trace(go.Scatter(x=path_index, y=lower, mode="lines", name="Forecast lower"))
        fig.add_trace(go.Scatter(x=path_index, y=upper, mode="lines", name="Forecast upper"))
        fig.add_hline(y=anchor, line_dash="dot", annotation_text="Scan anchor")
        visible_low = float(min(chart_df["low"].min(), np.min(lower)))
        visible_high = float(max(chart_df["high"].max(), np.max(upper)))
        fig.add_trace(go.Scatter(x=[path_end, path_end], y=[visible_low, visible_high], mode="lines", name="Scan expiry"))

    if action in ACTIONABLE and plan.get("entry_zone_low") is not None and plan.get("entry_zone_high") is not None:
        marker_price = (float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0
        fig.add_trace(
            go.Scatter(
                x=[chart_df.index[-1]],
                y=[marker_price],
                mode="markers+text",
                text=[action],
                textposition="top center",
                name="Unified signal",
            )
        )
        for key, label in [("stop", "Stop Loss"), ("tp1", "Take Profit 1"), ("tp2", "Take Profit 2")]:
            if plan.get(key) is not None:
                fig.add_hline(y=float(plan[key]), line_dash="dash", annotation_text=label)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def render_quant_validation(quant: dict) -> None:
    with st.expander("Quant Validation", expanded=False):
        if not quant.get("ready"):
            st.info(quant.get("reason", "Quant calibration is unavailable."))
            return
        selected = quant["selected_diagnostics"]
        mode_text = "Validated — quant gate active" if quant.get("quant_gate_active") else "Calibration mode — rule engine remains primary"
        st.caption(mode_text)
        cols = st.columns(6)
        cols[0].metric("Samples", selected["samples"])
        cols[1].metric("Validation", selected["validation_samples"])
        cols[2].metric("Skill", f"{selected['skill']:.3f}")
        cols[3].metric("Direction", fmt_percent(selected["directional_accuracy"]))
        cols[4].metric("Coverage 80", fmt_percent(selected["coverage_80"]))
        cols[5].metric("IC lower 95%", f"{selected['ic_lower_95']:.3f}")
        st.dataframe(pd.DataFrame(quant["candidate_diagnostics"]), use_container_width=True)


default_settings = load_yaml(ROOT / "config/default_settings.yaml")
presets = load_yaml(ROOT / "config/presets.yaml")
_initialize_session(default_settings)
settings_panel(default_settings, presets)
page = st.sidebar.radio("Page", ["Forecast Manager", "KC Squeeze", "Market Map", "Macro / News", "Settings", "Log Snapshot"])

settings = dict(st.session_state.applied_settings)
data_mode = str(st.session_state.applied_data_mode)
macro_bias = str(st.session_state.applied_macro_bias)
csv_bytes = st.session_state.applied_csv_bytes

st.title("XAU/USD Forecast Manager")
st.caption("One unified rule-and-quant scan. Apply settings first, then scan. No broker execution or auto-trading.")
scan_col, status_col = st.columns([1, 3])
with scan_col:
    scan_clicked = st.button("Scan Market", type="primary", use_container_width=True, key="scan_market")
with status_col:
    if st.session_state.scan_result is None:
        st.info("No active scan. Apply the sidebar settings, then press Scan Market.")
    else:
        completed_at = st.session_state.scan_result.get("scanned_at")
        if completed_at is None:
            st.warning("The last scan did not complete. Review the error below.")
        else:
            st.success(f"Last scan completed: {completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

if scan_clicked:
    with st.spinner("Running data, strategy, quant forecast, probability, expected value and veto modules..."):
        st.session_state.scan_result = run_scan(settings, macro_bias, data_mode, csv_bytes)
    st.rerun()

result = st.session_state.scan_result
if result is None:
    st.stop()
if result.get("error"):
    st.error(result["error"])
    feed_error = result.get("feed")
    if feed_error is not None and feed_error.warning:
        st.warning(feed_error.warning)
    st.stop()

settings = result["settings"]
feed = result["feed"]
bars = result["bars"]
kc = result["kc"]
market = result["market"]
regime = result["regime"]
macro = result["macro"]
scores = result["scores"]
plan = result["plan"]
bands = result["bands"]
quant = result["quant"]
status_snapshot: dict = {}
scan_age_seconds = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(result["scanned_at"])).total_seconds()
if result["data_mode"] == "Twelve Data" and quant.get("ready") and scan_age_seconds >= 60:
    status_snapshot = load_live_validity_snapshot(json.dumps(settings, sort_keys=True, default=str))
current_price = float(status_snapshot.get("price", market.price))
current_atr = float(status_snapshot.get("atr", market.atr))
current_regime = str(status_snapshot.get("regime", regime))
validity = scan_validity(
    quant,
    current_price=current_price,
    current_regime=current_regime,
    current_atr=current_atr,
)
veto = _effective_veto(result, validity)
active_plan = veto["final_action"] in ACTIONABLE
advisory_qty = result["advisory_qty"] if active_plan else 0.0

source_cols = st.columns(4)
source_cols[0].metric("Data source", feed.source)
source_cols[1].metric("Current price", fmt_price(current_price))
source_cols[2].metric("Rows loaded", f"{len(feed.bars):,}")
source_cols[3].metric("Rows after warm-up", f"{len(bars):,}")
if feed.warning:
    st.warning(feed.warning)

if page == "Forecast Manager":
    validity_primary = st.columns(4)
    validity_primary[0].metric("Scan status", validity["status"])
    validity_primary[1].metric("Current price", fmt_price(current_price))
    deviation = validity.get("price_deviation_sigma")
    validity_primary[2].metric("Path deviation", f"{deviation:.2f}σ" if deviation is not None else "N/A")
    validity_primary[3].metric("Remaining", fmt_duration(validity.get("remaining_seconds", 0)))
    validity_secondary = st.columns(4)
    validity_secondary[0].metric("Horizon", f"{quant.get('horizon_bars', 0)} bars")
    validity_secondary[1].metric(
        "Expiry",
        pd.Timestamp(quant["expiry_time"]).strftime("%Y-%m-%d %H:%M UTC") if quant.get("ready") else "N/A",
    )
    validity_secondary[2].metric("Quant mode", str(quant.get("mode", "unavailable")).title())
    half_life = quant.get("half_life_bars")
    validity_secondary[3].metric("Signal half-life", f"{half_life:.2f} bars" if half_life is not None else "N/A")
    if validity["status"] == "EXPIRED":
        reason_text = "; ".join(validity.get("reasons", []))
        st.error("EXPIRED — RESCAN REQUIRED" + (f": {reason_text}" if reason_text else ""))
    elif validity["status"] == "WEAKENING":
        st.warning("Scan edge is weakening. Rescan before treating the plan as actionable.")
    if status_snapshot.get("warning"):
        st.caption("Validity refresh warning: " + str(status_snapshot["warning"]))

    if quant.get("ready"):
        forecast_cols = st.columns(5)
        forecast_cols[0].metric("Anchor price", fmt_price(quant["anchor_price"]))
        forecast_cols[1].metric("Expected price", fmt_price(quant["expected_price"]))
        forecast_cols[2].metric("Lower forecast", fmt_price(quant["lower_price"]))
        forecast_cols[3].metric("Upper forecast", fmt_price(quant["upper_price"]))
        direction_probability = quant["probability_up"] if quant["expected_log_return"] >= 0 else quant["probability_down"]
        forecast_cols[4].metric("Direction probability", fmt_percent(direction_probability))
    else:
        st.info(quant.get("reason", "Quant calibration is unavailable; rule strategy remains active."))

    strategy_cols = st.columns(5)
    strategy_cols[0].metric("Final action", veto["final_action"])
    strategy_cols[1].metric("Rule action", veto.get("rule_action", result["rule_veto"]["final_action"]))
    strategy_cols[2].metric("Quality", veto["trade_quality"])
    strategy_cols[3].metric("Regime", regime)
    strategy_cols[4].metric("Rule confidence", scores["confidence"])

    score_cols = st.columns(4)
    score_cols[0].metric("Bull score", scores["bull_score"])
    score_cols[1].metric("Bear score", scores["bear_score"])
    score_cols[2].metric("Range position", f"{market.range_position_pct:.1f}%")
    score_cols[3].metric("Room ratio", f"{veto['room_ratio']:.2f}")
    st.info(result["summary"])
    st.caption(plan.get("note", ""))
    if veto["reasons"]:
        st.error("Veto reasons: " + "; ".join(veto["reasons"]))

    trade_cols = st.columns(6)
    trade_cols[0].metric("Entry low", fmt_price(active_value(plan, "entry_zone_low", veto["final_action"])))
    trade_cols[1].metric("Entry high", fmt_price(active_value(plan, "entry_zone_high", veto["final_action"])))
    trade_cols[2].metric("Stop Loss", fmt_price(active_value(plan, "stop", veto["final_action"])))
    trade_cols[3].metric("Take Profit 1", fmt_price(active_value(plan, "tp1", veto["final_action"])))
    trade_cols[4].metric("TP before SL", fmt_percent(quant.get("tp_before_sl_probability")))
    trade_cols[5].metric("Expected value", fmt_price(quant.get("expected_value")))
    extra_cols = st.columns(2)
    extra_cols[0].metric("Take Profit 2", fmt_price(active_value(plan, "tp2", veto["final_action"])))
    extra_cols[1].metric("Advisory units @ $10k", fmt_units(advisory_qty))

    st.subheader("Price, forecast path and strategy levels")
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan, quant), use_container_width=True)
    render_quant_validation(quant)

elif page == "KC Squeeze":
    st.subheader("KC Squeeze module")
    st.json(kc)
    columns = ["close", "trend_sma", "bb_upper", "bb_lower", "kc_upper", "kc_lower", "squeeze_on", "squeeze_fired", "kc_momentum"]
    st.dataframe(bars[[column for column in columns if column in bars.columns]].tail(30), use_container_width=True)
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan, quant), use_container_width=True)

elif page == "Market Map":
    st.subheader("Market location")
    st.json(market.to_dict())
    st.subheader("Expected bands")
    st.json(bands)
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan, quant), use_container_width=True)

elif page == "Macro / News":
    st.subheader("Macro context")
    st.json(macro)
    st.write("Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches.")

elif page == "Settings":
    st.subheader("Applied settings used by the last scan")
    settings_view = dict(settings)
    settings_view["data_mode"] = result["data_mode"]
    settings_view["macro_bias"] = result["macro_bias"]
    st.json(settings_view)
    st.download_button("Download settings YAML", yaml.safe_dump(settings_view, sort_keys=False), file_name="active_settings.yaml")

elif page == "Log Snapshot":
    row = {
        "scanned_at_utc": result["scanned_at"].isoformat(),
        "scan_status": validity["status"],
        "source": feed.source,
        "price": current_price,
        "regime": current_regime,
        "rule_action": veto.get("rule_action"),
        "final_action": veto["final_action"],
        "quant_mode": quant.get("mode"),
        "quant_horizon_bars": quant.get("horizon_bars"),
        "quant_expected_price": quant.get("expected_price"),
        "quant_lower_price": quant.get("lower_price"),
        "quant_upper_price": quant.get("upper_price"),
        "probability_up": quant.get("probability_up"),
        "price_deviation_sigma": validity.get("price_deviation_sigma"),
        "tp_before_sl_probability": quant.get("tp_before_sl_probability"),
        "expected_value": quant.get("expected_value"),
        "confidence": scores["confidence"],
        "quality": veto["trade_quality"],
        "veto_reasons": "; ".join(veto["reasons"]),
        "entry_zone_low": active_value(plan, "entry_zone_low", veto["final_action"]),
        "entry_zone_high": active_value(plan, "entry_zone_high", veto["final_action"]),
        "stop_loss": active_value(plan, "stop", veto["final_action"]),
        "take_profit_1": active_value(plan, "tp1", veto["final_action"]),
        "take_profit_2": active_value(plan, "tp2", veto["final_action"]),
    }
    snap = pd.DataFrame([row])
    st.dataframe(snap, use_container_width=True)
    st.download_button("Download snapshot CSV", snap.to_csv(index=False), file_name="xauusd_forecast_snapshot.csv")
