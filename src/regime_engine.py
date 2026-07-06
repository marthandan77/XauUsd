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
        momentum = float(row.get("kc_momentum", 0) or 0)
        previous_momentum = float(row.get("kc_momentum_prev", 0) or 0)
        trend_sma = float(row.get("trend_sma", ema_slow) or ema_slow)

        if squeeze_fired and momentum > 0 and momentum > previous_momentum and price > trend_sma:
            return "bullish_squeeze_breakout"
        if squeeze_fired and momentum < 0 and momentum < previous_momentum and price < trend_sma:
            return "bearish_squeeze_breakout"
        if squeeze_on:
            return "compression"

    if ema_fast > ema_slow and price > ema_fast and adx >= adx_min:
        return "bull_trend"
    if ema_fast < ema_slow and price < ema_fast and adx >= adx_min:
        return "bear_trend"
    return "range"
