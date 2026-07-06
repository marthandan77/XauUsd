from __future__ import annotations

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

st.set_page_config(page_title="XAU/USD Forecast Manager", layout="wide")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _select_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def settings_panel(settings: dict, presets: dict) -> dict:
    st.sidebar.header("Forecast controls")
    if presets:
        selected = st.sidebar.selectbox("Preset", list(presets.keys()), index=0)
        if st.sidebar.button("Load preset"):
            settings.update(presets[selected])

    interval_options = ["15m", "30m", "1h", "4h", "1d"]
    period_options = ["7d", "30d", "60d", "6mo", "1y", "2y"]
    settings["price_interval"] = st.sidebar.selectbox(
        "Timeframe", interval_options, index=_select_index(interval_options, str(settings.get("price_interval", "15m")))
    )
    settings["price_period"] = st.sidebar.selectbox(
        "Lookback period", period_options, index=_select_index(period_options, str(settings.get("price_period", "30d")))
    )

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
    settings["risk_per_trade_pct"] = st.sidebar.slider("Advisory risk %", 0.1, 2.0, float(settings.get("risk_per_trade_pct", 0.5)), 0.1)
    settings["long_plans_enabled"] = st.sidebar.toggle("Enable long plans", bool(settings.get("long_plans_enabled", True)))
    settings["short_plans_enabled"] = st.sidebar.toggle("Enable short plans", bool(settings.get("short_plans_enabled", False)))
    settings["show_bollinger_bands"] = st.sidebar.toggle("Show Bollinger Bands", bool(settings.get("show_bollinger_bands", True)))
    settings["show_keltner_channels"] = st.sidebar.toggle("Show Keltner Channels", bool(settings.get("show_keltner_channels", True)))
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
    if action in {"BUY PLAN", "SELL PLAN"}:
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


settings = load_yaml(ROOT / "config/default_settings.yaml")
presets = load_yaml(ROOT / "config/presets.yaml")
settings = settings_panel(settings, presets)
macro_bias = st.sidebar.selectbox("Macro bias", ["mixed", "supportive", "restrictive"], index=0)
feed = load_bars(settings)

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
scores = score_forecast(latest_row, market, regime, macro, settings)
if kc["state"] != "insufficient_data":
    scores["kc_state"] = kc["state"]
    scores["kc_reason"] = kc["reason"]
action = choose_action(scores, settings)
plan = build_trade_plan(action, market, settings)
bands = price_bands(market, regime)
veto = apply_veto(action, plan, market, regime, macro, settings)
summary = explain(regime, scores, veto, market, macro)

advisory_qty = 0.0
if veto["final_action"] in {"BUY PLAN", "SELL PLAN"}:
    advisory_qty = advisory_position_size(
        account_equity=10000,
        risk_pct=float(settings.get("risk_per_trade_pct", 0.5)) / 100.0,
        entry=(float(plan["entry_zone_low"]) + float(plan["entry_zone_high"])) / 2.0,
        stop=float(plan["stop"]),
    )

st.title("XAU/USD Forecast Manager")
st.caption("Advisory dashboard only. No broker execution. No auto-trading. No backtesting. Veto first, signal second.")
page = st.sidebar.radio("Page", ["Forecast Manager", "KC Squeeze", "Market Map", "Macro / News", "Settings", "Log Snapshot"])

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
    p5, p6 = st.columns(2)
    p5.metric("Target 2", f"{plan['tp2']:,.2f}")
    p6.metric("Advisory units @ $10k equity", f"{advisory_qty:,.2f}")
    st.subheader("Price and indicators")
    st.plotly_chart(price_chart(bars, settings, veto["final_action"], plan), use_container_width=True)

elif page == "KC Squeeze":
    st.subheader("KC Squeeze module")
    st.json(kc)
    cols = ["close", "trend_sma", "bb_upper", "bb_lower", "kc_upper", "kc_lower", "squeeze_on", "squeeze_fired", "kc_momentum"]
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

elif page == "Log Snapshot":
    row = {
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
        "entry_zone_low": plan["entry_zone_low"],
        "entry_zone_high": plan["entry_zone_high"],
        "guard_level": plan["stop"],
        "target_1": plan["tp1"],
        "target_2": plan["tp2"],
        "advisory_units_10k": advisory_qty,
    }
    snap = pd.DataFrame([row])
    st.dataframe(snap, use_container_width=True)
    st.download_button("Download snapshot CSV", snap.to_csv(index=False), file_name="xauusd_forecast_snapshot.csv")

if feed.warning:
    st.warning(feed.warning)
