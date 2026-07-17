from __future__ import annotations

import base64
import json
import math
from io import BytesIO
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


def _select_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def _sample_freq(settings: dict) -> str:
    return {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}.get(
        str(settings.get("price_interval", "15m")), "15min"
    )


def _sample_fallback(settings: dict, warning: str) -> FeedResult:
    if not bool(settings.get("sample_data_enabled", True)):
        return FeedResult(pd.DataFrame(), "none", warning)
    return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", warning)


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


def active_value(plan: dict, key: str, final_action: str):
    if final_action not in ACTIONABLE:
        return None
    return plan.get(key)


def _draft_key(setting_key: str) -> str:
    return f"draft_{setting_key}"


def _encode_persisted_config(settings: dict, data_mode: str, macro_bias: str) -> str:
    payload = {
        "settings": settings,
        "data_mode": data_mode,
        "macro_bias": macro_bias,
    }
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
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_persisted_config(settings: dict, data_mode: str, macro_bias: str) -> None:
    st.query_params["config"] = _encode_persisted_config(settings, data_mode, macro_bias)


def _clear_app_state() -> None:
    st.query_params.clear()
    managed_keys = {
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
        if key in managed_keys or key.startswith("draft_"):
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
        saved_mode = str(persisted.get("data_mode", "Twelve Data"))
        st.session_state.applied_data_mode = saved_mode if saved_mode in DATA_MODES else "Twelve Data"
    if "applied_macro_bias" not in st.session_state:
        saved_bias = str(persisted.get("macro_bias", "mixed"))
        st.session_state.applied_macro_bias = saved_bias if saved_bias in MACRO_BIASES else "mixed"
    if "applied_csv_bytes" not in st.session_state:
        st.session_state.applied_csv_bytes = None
    if "applied_csv_name" not in st.session_state:
        st.session_state.applied_csv_name = None
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = None
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
        apply_clicked = apply_col.form_submit_button(
            "Apply Settings", type="primary", use_container_width=True
        )

        st.subheader("Data")
        data_mode = st.radio("Data mode", DATA_MODES, key="draft_data_mode")
        symbol = st.text_input(
            "Twelve Data symbol",
            key="draft_twelve_data_symbol",
            disabled=data_mode != "Twelve Data",
        )
        uploaded = st.file_uploader(
            "Upload OHLC CSV",
            type=["csv"],
            key="draft_uploaded_csv",
            disabled=data_mode != "CSV upload",
        )
        macro_bias = st.selectbox("Macro bias", MACRO_BIASES, key="draft_macro_bias")

        st.subheader("Forecast")
        price_interval = st.selectbox("Timeframe", INTERVAL_OPTIONS, key="draft_price_interval")
        price_period = st.selectbox("Lookback period", PERIOD_OPTIONS, key="draft_price_period")
        buy_threshold = st.slider("Bull threshold", 50, 95, key="draft_buy_threshold")
        sell_threshold = st.slider("Bear threshold", 50, 95, key="draft_sell_threshold")
        wait_threshold = st.slider("Watch threshold", 40, 90, key="draft_wait_threshold")
        atr_multiplier = st.slider(
            "ATR guard multiplier", 0.8, 2.5, step=0.1, key="draft_atr_multiplier"
        )
        min_reward_risk = st.slider(
            "Minimum room ratio", 1.0, 3.0, step=0.1, key="draft_min_reward_risk"
        )
        middle_range_lower_pct = st.slider(
            "Middle zone lower %", 10, 49, key="draft_middle_range_lower_pct"
        )
        middle_range_upper_pct = st.slider(
            "Middle zone upper %", 51, 90, key="draft_middle_range_upper_pct"
        )

        st.subheader("KC Squeeze and risk")
        kc_squeeze_enabled = st.toggle(
            "Enable KC Squeeze module", key="draft_kc_squeeze_enabled"
        )
        bb_length = st.slider("BB length", 10, 60, key="draft_bb_length")
        bb_mult = st.slider("BB multiplier", 1.0, 3.5, step=0.1, key="draft_bb_mult")
        kc_length = st.slider("KC length", 10, 60, key="draft_kc_length")
        kc_mult = st.slider("KC multiplier", 1.0, 3.5, step=0.1, key="draft_kc_mult")
        trend_length = st.slider("Trend SMA length", 50, 300, key="draft_trend_length")
        atr_period = st.slider("ATR period", 5, 50, key="draft_atr_period")
        atr_stop_multiplier = st.slider(
            "ATR stop multiplier", 1.0, 5.0, step=0.1, key="draft_atr_stop_multiplier"
        )
        atr_tp_multiplier = st.slider(
            "ATR take-profit multiplier", 1.0, 10.0, step=0.1, key="draft_atr_tp_multiplier"
        )
        risk_per_trade_pct = st.slider(
            "Advisory risk %", 0.1, 2.0, step=0.1, key="draft_risk_per_trade_pct"
        )
        long_plans_enabled = st.toggle("Enable long plans", key="draft_long_plans_enabled")
        short_plans_enabled = st.toggle("Enable short plans", key="draft_short_plans_enabled")
        show_bollinger_bands = st.toggle(
            "Show Bollinger Bands", key="draft_show_bollinger_bands"
        )
        show_keltner_channels = st.toggle(
            "Show Keltner Channels", key="draft_show_keltner_channels"
        )
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
        "Applied settings are saved in the app URL and survive browser refreshes. Draft changes do not affect a scan until Apply Settings is pressed."
    )


def load_bars(settings: dict, data_mode: str, csv_bytes: bytes | None) -> FeedResult:
    if data_mode == "CSV upload":
        if not csv_bytes:
            return _sample_fallback(settings, "CSV not uploaded; sample data used")
        feed = load_csv(BytesIO(csv_bytes))
        if not feed.bars.empty:
            return feed
        return _sample_fallback(settings, f"{feed.warning}; sample data used")

    if data_mode == "Twelve Data":
        feed = load_live_bars(settings)
        if not feed.bars.empty:
            return feed
        return _sample_fallback(settings, f"{feed.warning}; sample data used")

    return FeedResult(make_sample_bars(freq=_sample_freq(settings)), "sample", "sample data selected")


def run_scan(settings: dict, macro_bias: str, data_mode: str, csv_bytes: bytes | None) -> dict:
    feed = load_bars(settings, data_mode, csv_bytes)
    if feed.bars.empty:
        return {
            "error": "No market data is available. Check the data source, API secret, or CSV file.",
            "feed": feed,
        }

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
    veto = apply_veto(action, plan, market, regime, macro, settings)
    summary = explain(regime, scores, veto, market, macro)
    active_plan = veto["final_action"] in ACTIONABLE

    advisory_qty = 0.0
    if (
        active_plan
        and plan.get("entry_zone_low") is not None
        and plan.get("entry_zone_high") is not None
        and plan.get("stop") is not None
    ):
        advisory_qty = advisory_position_size(
            account_equity=10000,
            risk_pct=float(settings.get("risk_per_trade_pct", 0.5)) / 100.0,
            entry=(float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0,
            stop=float(plan["stop"]),
        )

    return {
        "error": "",
        "scanned_at": pd.Timestamp.now(tz="UTC"),
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
        "veto": veto,
        "summary": summary,
        "active_plan": active_plan,
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


default_settings = load_yaml(ROOT / "config/default_settings.yaml")
presets = load_yaml(ROOT / "config/presets.yaml")
_initialize_session(default_settings)
settings_panel(default_settings, presets)

page = st.sidebar.radio(
    "Page",
    ["Forecast Manager", "KC Squeeze", "Market Map", "Macro / News", "Settings", "Log Snapshot"],
)

settings = dict(st.session_state.applied_settings)
data_mode = str(st.session_state.applied_data_mode)
macro_bias = str(st.session_state.applied_macro_bias)
csv_bytes = st.session_state.applied_csv_bytes

st.title("XAU/USD Forecast Manager")
st.caption("Advisory dashboard only. Apply settings first, then scan. No broker execution or auto-trading.")

scan_col, status_col = st.columns([1, 3])
with scan_col:
    scan_clicked = st.button("Scan Market", type="primary", use_container_width=True, key="scan_market")
with status_col:
    if st.session_state.scan_result is None:
        st.info("No active scan. Apply the sidebar settings, then press Scan Market.")
    else:
        scanned_at = st.session_state.scan_result.get("scanned_at")
        if scanned_at is not None:
            st.success(f"Last scan completed: {scanned_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

if scan_clicked:
    with st.spinner("Scanning data and running all strategy modules..."):
        st.session_state.scan_result = run_scan(settings, macro_bias, data_mode, csv_bytes)
    st.rerun()

result = st.session_state.scan_result
if result is None:
    st.stop()

if result.get("error"):
    st.error(result["error"])
    feed = result.get("feed")
    if feed is not None and feed.warning:
        st.warning(feed.warning)
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
veto = result["veto"]
summary = result["summary"]
active_plan = result["active_plan"]
advisory_qty = result["advisory_qty"]

source_cols = st.columns(4)
source_cols[0].metric("Data source", feed.source)
source_cols[1].metric("Latest price", fmt_price(market.price))
source_cols[2].metric("Rows loaded", f"{len(feed.bars):,}")
source_cols[3].metric("Rows after warm-up", f"{len(bars):,}")
if feed.warning:
    st.warning(feed.warning)

if page == "Forecast Manager":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final action", veto["final_action"])
    c2.metric("Quality", veto["trade_quality"])
    c3.metric("Regime", regime)
    c4.metric("Confidence", scores["confidence"])
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Bull score", scores["bull_score"])
    s2.metric("Bear score", scores["bear_score"])
    s3.metric("Range position", f"{market.range_position_pct:.1f}%")
    s4.metric("Room ratio", f"{veto['room_ratio']:.2f}")
    st.info(summary)
    st.caption(plan.get("note", ""))
    if veto["reasons"]:
        st.error("Veto reasons: " + "; ".join(veto["reasons"]))
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Entry zone low", fmt_price(active_value(plan, "entry_zone_low", veto["final_action"])))
    p2.metric("Entry zone high", fmt_price(active_value(plan, "entry_zone_high", veto["final_action"])))
    p3.metric("Guard level", fmt_price(active_value(plan, "stop", veto["final_action"])))
    p4.metric("Target 1", fmt_price(active_value(plan, "tp1", veto["final_action"])))
    p5, p6 = st.columns(2)
    p5.metric("Target 2", fmt_price(active_value(plan, "tp2", veto["final_action"])))
    p6.metric("Advisory units @ $10k equity", fmt_units(advisory_qty))
    st.subheader("Price and indicators")
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan), use_container_width=True)

elif page == "KC Squeeze":
    st.subheader("KC Squeeze module")
    st.json(kc)
    cols = [
        "close",
        "trend_sma",
        "bb_upper",
        "bb_lower",
        "kc_upper",
        "kc_lower",
        "squeeze_on",
        "squeeze_fired",
        "kc_momentum",
    ]
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
    st.write("Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches.")

elif page == "Settings":
    st.subheader("Applied settings used by the last scan")
    settings_view = dict(settings)
    settings_view["data_mode"] = result["data_mode"]
    settings_view["macro_bias"] = result["macro_bias"]
    if st.session_state.applied_csv_name:
        settings_view["csv_file"] = st.session_state.applied_csv_name
    st.json(settings_view)
    st.download_button(
        "Download settings YAML",
        yaml.safe_dump(settings_view, sort_keys=False),
        file_name="active_settings.yaml",
    )

elif page == "Log Snapshot":
    row = {
        "scanned_at_utc": result["scanned_at"].isoformat(),
        "source": feed.source,
        "warning": feed.warning,
        "price": market.price,
        "regime": regime,
        "kc_state": kc["state"],
        "bias": scores["bias"],
        "confidence": scores["confidence"],
        "final_action": veto["final_action"],
        "quality": veto["trade_quality"],
        "veto_reasons": "; ".join(veto["reasons"]),
        "entry_zone_low": active_value(plan, "entry_zone_low", veto["final_action"]),
        "entry_zone_high": active_value(plan, "entry_zone_high", veto["final_action"]),
        "guard_level": active_value(plan, "stop", veto["final_action"]),
        "target_1": active_value(plan, "tp1", veto["final_action"]),
        "target_2": active_value(plan, "tp2", veto["final_action"]),
        "advisory_units_10k": advisory_qty if active_plan else None,
        "plan_note": plan.get("note", ""),
    }
    snap = pd.DataFrame([row])
    st.dataframe(snap, use_container_width=True)
    st.download_button(
        "Download snapshot CSV",
        snap.to_csv(index=False),
        file_name="xauusd_forecast_snapshot.csv",
    )
