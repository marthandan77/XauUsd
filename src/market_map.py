from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


BARS_PER_DAY = {
    "5m": 288,
    "15m": 96,
    "30m": 48,
    "1h": 24,
    "4h": 6,
    "1d": 1,
}


@dataclass
class MarketMap:
    price: float
    atr: float
    high_7d: float
    low_7d: float
    support: float
    resistance: float
    swing_high: float
    swing_low: float
    vwap: float
    range_position_pct: float
    near_support: bool
    near_resistance: bool
    near_vwap: bool
    middle_range: bool
    atr_shock: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _period_days(settings: dict) -> int:
    period = str(settings.get("price_period", f"{settings.get('lookback_days', 7)}d"))
    try:
        if period.endswith("d"):
            return max(int(period[:-1]), 1)
        if period.endswith("mo"):
            return max(int(period[:-2]) * 30, 1)
        if period.endswith("y"):
            return max(int(period[:-1]) * 365, 1)
    except Exception:
        pass
    return max(int(settings.get("lookback_days", 7)), 1)


def _bars_per_day(settings: dict) -> int:
    return BARS_PER_DAY.get(str(settings.get("price_interval", "15m")), 96)


def build_market_map(df: pd.DataFrame, settings: dict) -> MarketMap:
    if df.empty:
        raise ValueError("cannot build market map from empty data")
    latest = df.iloc[-1]
    requested_window = _period_days(settings) * _bars_per_day(settings)
    window = df.tail(max(80, requested_window))
    price = float(latest["close"])
    high_7d = float(window["high"].max())
    low_7d = float(window["low"].min())
    atr = float(latest.get("atr", 0) or 0)
    if atr <= 0:
        atr = max((high_7d - low_7d) / 50.0, 0.01)
    recent = window.tail(20)
    vwap = float(latest.get("vwap", price))
    range_pos = 50.0 if high_7d == low_7d else ((price - low_7d) / (high_7d - low_7d)) * 100.0
    tol = float(settings.get("support_resistance_tolerance_atr", 0.5)) * atr
    vwap_tol = float(settings.get("vwap_tolerance_atr", 0.35)) * atr
    middle = float(settings.get("middle_range_lower_pct", 35)) <= range_pos <= float(settings.get("middle_range_upper_pct", 65))
    tr_mean = float(window["tr"].tail(50).mean()) if "tr" in window.columns else atr
    shock = float(latest.get("tr", 0) or 0) > float(settings.get("atr_shock_multiple", 2.2)) * max(tr_mean, 0.01)
    return MarketMap(
        price=price,
        atr=atr,
        high_7d=high_7d,
        low_7d=low_7d,
        support=low_7d,
        resistance=high_7d,
        swing_high=float(recent["high"].max()),
        swing_low=float(recent["low"].min()),
        vwap=vwap,
        range_position_pct=float(range_pos),
        near_support=abs(price - low_7d) <= tol,
        near_resistance=abs(high_7d - price) <= tol,
        near_vwap=abs(price - vwap) <= vwap_tol,
        middle_range=bool(middle),
        atr_shock=bool(shock),
    )
