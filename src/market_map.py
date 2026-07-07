from __future__ import annotations

from dataclasses import asdict, dataclass

import math

import pandas as pd


def _bars_per_day(interval: str) -> int:
    return {
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }.get(str(interval), 96)


def _finite(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


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
    previous_day_high: float | None = None
    previous_day_low: float | None = None
    asia_high: float | None = None
    asia_low: float | None = None
    london_high: float | None = None
    london_low: float | None = None
    round_support: float | None = None
    round_resistance: float | None = None
    support_label: str = "lookback_low"
    resistance_label: str = "lookback_high"
    liquidity_sweep_up: bool = False
    liquidity_sweep_down: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out[out.index.notna()].copy()


def _previous_day_levels(df: pd.DataFrame) -> tuple[float | None, float | None]:
    work = _ensure_datetime_index(df)
    if work.empty:
        return None, None
    daily = work.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    if len(daily) < 2:
        return None, None
    previous = daily.iloc[-2]
    return _finite(previous.get("high")), _finite(previous.get("low"))


def _session_levels(df: pd.DataFrame, start_hour: int, end_hour: int) -> tuple[float | None, float | None]:
    work = _ensure_datetime_index(df)
    if work.empty:
        return None, None
    current_date = work.index[-1].date()
    session = work[work.index.date == current_date]
    if session.empty:
        return None, None
    if start_hour < end_hour:
        session = session[(session.index.hour >= start_hour) & (session.index.hour < end_hour)]
    else:
        session = session[(session.index.hour >= start_hour) | (session.index.hour < end_hour)]
    if session.empty:
        return None, None
    return _finite(session["high"].max()), _finite(session["low"].min())


def _round_levels(price: float, step: float) -> tuple[float, float]:
    step = max(float(step), 0.01)
    support = math.floor(price / step) * step
    resistance = math.ceil(price / step) * step
    if resistance <= price:
        resistance += step
    if support >= price:
        support -= step
    return support, resistance


def _nearest_structural_levels(price: float, levels: list[tuple[str, float | None]], fallback_support: float, fallback_resistance: float) -> tuple[float, str, float, str]:
    clean = [(label, float(value)) for label, value in levels if value is not None and math.isfinite(float(value))]
    supports = [(label, value) for label, value in clean if value < price]
    resistances = [(label, value) for label, value in clean if value > price]
    if supports:
        support_label, support = max(supports, key=lambda item: item[1])
    else:
        support_label, support = "fallback_support", float(fallback_support)
    if resistances:
        resistance_label, resistance = min(resistances, key=lambda item: item[1])
    else:
        resistance_label, resistance = "fallback_resistance", float(fallback_resistance)
    return support, support_label, resistance, resistance_label


def build_market_map(df: pd.DataFrame, settings: dict) -> MarketMap:
    if df.empty:
        raise ValueError("cannot build market map from empty data")
    latest = df.iloc[-1]
    bars_per_day = _bars_per_day(str(settings.get("price_interval", "15m")))
    window = df.tail(max(80, int(settings.get("lookback_days", 7)) * bars_per_day))
    price = float(latest["close"])
    high_7d = float(window["high"].max())
    low_7d = float(window["low"].min())
    atr = float(latest.get("atr", 0) or 0)
    if atr <= 0:
        atr = max((high_7d - low_7d) / 50.0, 0.01)

    recent = window.tail(20)
    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())
    previous_day_high, previous_day_low = _previous_day_levels(df)
    asia_high, asia_low = _session_levels(df, 0, 8)
    london_high, london_low = _session_levels(df, 7, 16)
    round_step = float(settings.get("round_level_step", 10.0))
    round_support, round_resistance = _round_levels(price, round_step)

    level_candidates = [
        ("lookback_low", low_7d),
        ("lookback_high", high_7d),
        ("swing_low", swing_low),
        ("swing_high", swing_high),
        ("previous_day_low", previous_day_low),
        ("previous_day_high", previous_day_high),
        ("asia_low", asia_low),
        ("asia_high", asia_high),
        ("london_low", london_low),
        ("london_high", london_high),
        ("round_support", round_support),
        ("round_resistance", round_resistance),
    ]
    fallback_support = min(low_7d, price - atr)
    fallback_resistance = max(high_7d, price + atr)
    support, support_label, resistance, resistance_label = _nearest_structural_levels(
        price, level_candidates, fallback_support, fallback_resistance
    )

    vwap = float(latest.get("vwap", price))
    range_pos = 50.0 if high_7d == low_7d else ((price - low_7d) / (high_7d - low_7d)) * 100.0
    tol = float(settings.get("support_resistance_tolerance_atr", 0.5)) * atr
    vwap_tol = float(settings.get("vwap_tolerance_atr", 0.35)) * atr
    middle = float(settings.get("middle_range_lower_pct", 35)) <= range_pos <= float(settings.get("middle_range_upper_pct", 65))
    tr_mean = float(window["tr"].tail(50).mean()) if "tr" in window.columns else atr
    shock = float(latest.get("tr", 0) or 0) > float(settings.get("atr_shock_multiple", 2.2)) * max(tr_mean, 0.01)

    prev_hi = previous_day_high if previous_day_high is not None else high_7d
    prev_lo = previous_day_low if previous_day_low is not None else low_7d
    liquidity_sweep_up = bool(float(latest["high"]) > float(prev_hi) and price < float(prev_hi)) if prev_hi is not None else False
    liquidity_sweep_down = bool(float(latest["low"]) < float(prev_lo) and price > float(prev_lo)) if prev_lo is not None else False

    return MarketMap(
        price=price,
        atr=atr,
        high_7d=high_7d,
        low_7d=low_7d,
        support=float(support),
        resistance=float(resistance),
        swing_high=swing_high,
        swing_low=swing_low,
        vwap=vwap,
        range_position_pct=float(range_pos),
        near_support=abs(price - float(support)) <= tol,
        near_resistance=abs(float(resistance) - price) <= tol,
        near_vwap=abs(price - vwap) <= vwap_tol,
        middle_range=bool(middle),
        atr_shock=bool(shock),
        previous_day_high=previous_day_high,
        previous_day_low=previous_day_low,
        asia_high=asia_high,
        asia_low=asia_low,
        london_high=london_high,
        london_low=london_low,
        round_support=round_support,
        round_resistance=round_resistance,
        support_label=support_label,
        resistance_label=resistance_label,
        liquidity_sweep_up=liquidity_sweep_up,
        liquidity_sweep_down=liquidity_sweep_down,
    )
