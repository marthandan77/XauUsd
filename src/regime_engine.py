from __future__ import annotations

import pandas as pd

from .market_map import MarketMap


def classify_regime(df: pd.DataFrame, market: MarketMap, settings: dict) -> str:
    row = df.iloc[-1]
    price = float(row["close"])
    ema_fast = float(row.get("ema_fast", price))
    ema_slow = float(row.get("ema_slow", price))
    adx = float(row.get("adx", 0))
    adx_min = float(settings.get("adx_minimum", 20))

    if market.atr_shock:
        return "shock"

    if bool(settings.get("kc_squeeze_enabled", True)):
        squeeze_on = bool(row.get("squeeze_on", False))
        squeeze_fired = bool(row.get("squeeze_fired", False))
        squeeze_recent = bool(row.get("squeeze_recent", False))
        bullish_confirmed = bool(row.get("release_bullish_confirmed", False))
        bearish_confirmed = bool(row.get("release_bearish_confirmed", False))
        release_chase_risk = bool(row.get("release_chase_risk", False))

        if squeeze_on:
            return "compression"
        if squeeze_recent and bullish_confirmed:
            return "bullish_release_overextended" if release_chase_risk else "bullish_release_confirmed"
        if squeeze_recent and bearish_confirmed:
            return "bearish_release_overextended" if release_chase_risk else "bearish_release_confirmed"
        if squeeze_fired:
            return "squeeze_release_now"
        if squeeze_recent:
            return "squeeze_release_recent"

    if ema_fast > ema_slow and price > ema_fast and adx >= adx_min:
        return "bull_trend"
    if ema_fast < ema_slow and price < ema_fast and adx >= adx_min:
        return "bear_trend"
    return "range"
