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
    if ema_fast > ema_slow and price > ema_fast and adx >= adx_min:
        return "bull_trend"
    if ema_fast < ema_slow and price < ema_fast and adx >= adx_min:
        return "bear_trend"
    return "range"
