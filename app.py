from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.ai_explainer import explain
from src.data_feed import FeedResult, load_csv, load_live_bars, make_sample_bars
from src.forecast_engine import choose_action, score_forecast
from src.indicators import add_indicators
from src.macro_engine import macro_context
from src.market_map import build_market_map
from src.range_model import price_bands
from src.regime_engine import classify_regime
from src.trade_plan import build_trade_plan
from src.veto_engine import apply_veto

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="XAU/USD Forecast Manager", layout="wide")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def settings_panel(settings: dict, presets: dict) -> dict:
    st.sidebar.header("Forecast controls")
    if presets:
        selected = st.sidebar.selectbox("Preset", list(presets.keys()), index=0)
        if st.sidebar.button("Load preset"):
            settings.update(presets[selected])
    settings["buy_threshold"] = st.sidebar.slider("Bull threshold", 50, 95, int(settings.get("buy_threshold", 78)))
    settings["sell_threshold"] = st.sidebar.slider("Bear threshold", 50, 95, int(settings.get("sell_threshold", 78)))
    settings["wait_threshold"] = st.sidebar.slider("Watch threshold", 40, 90, int(settings.get("wait_threshold", 60)))
    settings["atr_multiplier"] = st.sidebar.slider("ATR guard multiplier", 0.8, 2.5, float(settings.get("atr_multiplier", 1.3)), 0.1)
    settings["min_reward_risk"] = st.sidebar.slider("Minimum room ratio", 1.0, 3.0, float(settings.get("min_reward_risk", 1.5)), 0.1)
    settings["middle_range_lower_pct"] = st.sidebar.slider("Middle zone lower %", 10, 49, int(settings.get("middle_range_lower_pct", 35)))
    settings["middle_range_upper_pct"] = st.sidebar.slider("Middle zone upper %", 51, 90, int(settings.get("middle_range_upper_pct", 65)))
    settings["news_block_manual"] = st.sidebar.toggle("Manual event block", bool(settings.get("news_block_manual", False)))
    return settings


def load_bars(settings: dict) -> FeedResult:
    st.sidebar.header("Data")
    data_mode = st.sidebar.radio("Data mode", ["Live yfinance", "CSV upload", "Sample"], index=0)
    if data_mode == "CSV upload":
        uploaded = st.sidebar.file_uploader("Upload OHLC CSV", type=["csv"])
        if uploaded is not None:
            return load_csv(uploaded)
        return FeedResult(make_sample_bars(), "sample", "CSV not uploaded; sample data used")
    if data_mode == "Live yfinance":
        feed = load_live_bars(settings)
        if not feed.bars.empty:
            return feed
        return FeedResult(make_sample_bars(), "sample", "live feed unavailable; sample data used. " + feed.warning)
    return FeedResult(make_sample_bars(), "sample", "sample data")


settings = load_yaml(ROOT / "config/default_settings.yaml")
presets = load_yaml(ROOT / "config/presets.yaml")
settings = settings_panel(settings, presets)
macro_bias = st.sidebar.selectbox("Macro bias", ["mixed", "supportive", "restrictive"], index=0)
feed = load_bars(settings)

bars = add_indicators(feed.bars, settings).dropna().copy()
market = build_market_map(bars, settings)
regime = classify_regime(bars, market, settings)
macro = macro_context(macro_bias, bool(settings.get("news_block_manual", False)))
scores = score_forecast(bars.iloc[-1].to_dict(), market, regime, macro, settings)
action = choose_action(scores, settings)
plan = build_trade_plan(action, market, settings)
bands = price_bands(market, regime)
veto = apply_veto(action, plan, market, regime, macro, settings)
summary = explain(regime, scores, veto, market, macro)

st.title("XAU/USD Forecast Manager")
st.caption("Manual dashboard. No auto-trading. No auto-learning. Veto first, signal second.")
page = st.sidebar.radio("Page", ["Forecast Manager", "Market Map", "Macro / News", "Settings", "Log Snapshot"])

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
    if veto["reasons"]:
        st.error("Veto reasons: " + "; ".join(veto["reasons"]))
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Entry zone low", f"{plan['entry_zone_low']:,.2f}")
    p2.metric("Entry zone high", f"{plan['entry_zone_high']:,.2f}")
    p3.metric("Guard level", f"{plan['stop']:,.2f}")
    p4.metric("Target 1", f"{plan['tp1']:,.2f}")
    st.subheader("Price and indicators")
    chart_df = bars[["close", "ema_fast", "ema_slow", "vwap"]].tail(160)
    st.line_chart(chart_df)

elif page == "Market Map":
    st.subheader("Market location")
    st.json(market.to_dict())
    st.subheader("Expected bands")
    st.json(bands)
    st.line_chart(bars[["close", "ema_fast", "ema_slow", "vwap"]].tail(220))

elif page == "Macro / News":
    st.subheader("Macro context")
    st.json(macro)
    st.write("Use the manual event block around CPI, NFP, PCE, FOMC and major Fed speeches. API calendar can be added later.")

elif page == "Settings":
    st.subheader("Active settings")
    st.json(settings)
    st.download_button("Download settings YAML", yaml.safe_dump(settings, sort_keys=False), file_name="active_settings.yaml")

elif page == "Log Snapshot":
    row = {
        "source": feed.source,
        "warning": feed.warning,
        "price": market.price,
        "regime": regime,
        "bias": scores["bias"],
        "confidence": scores["confidence"],
        "final_action": veto["final_action"],
        "quality": veto["trade_quality"],
        "veto_reasons": "; ".join(veto["reasons"]),
        "entry_zone_low": plan["entry_zone_low"],
        "entry_zone_high": plan["entry_zone_high"],
        "guard_level": plan["stop"],
        "target_1": plan["tp1"],
        "target_2": plan["tp2"],
    }
    snap = pd.DataFrame([row])
    st.dataframe(snap, use_container_width=True)
    st.download_button("Download snapshot CSV", snap.to_csv(index=False), file_name="xauusd_forecast_snapshot.csv")

if feed.warning:
    st.warning(feed.warning)
