from __future__ import annotations

from dataclasses import asdict, dataclass

import math

import pandas as pd


@dataclass
class ExhaustionContext:
    state: str
    block_buy: bool
    block_sell: bool
    bull_exhaustion: bool
    bear_exhaustion: bool
    liquidity_sweep_up: bool
    liquidity_sweep_down: bool
    rsi: float | None
    kc_momentum: float | None
    range_position_pct: float | None
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _finite(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except Exception:
        return default
    return number if math.isfinite(number) else default


def exhaustion_context(bars: pd.DataFrame, market, settings: dict) -> ExhaustionContext:
    if bars is None or bars.empty or market is None:
        return ExhaustionContext(
            state="unknown",
            block_buy=False,
            block_sell=False,
            bull_exhaustion=False,
            bear_exhaustion=False,
            liquidity_sweep_up=False,
            liquidity_sweep_down=False,
            rsi=None,
            kc_momentum=None,
            range_position_pct=None,
            warnings=["No bars available for exhaustion check."],
        )

    row = bars.iloc[-1]
    previous = bars.iloc[-2] if len(bars) >= 2 else row
    rsi = _finite(row.get("rsi"), 50.0)
    kc_momentum = _finite(row.get("kc_momentum"), 0.0)
    kc_previous = _finite(previous.get("kc_momentum"), kc_momentum or 0.0)
    range_pos = _finite(getattr(market, "range_position_pct", None), 50.0)
    momentum_falling = (kc_momentum or 0.0) < (kc_previous or 0.0)
    momentum_rising = (kc_momentum or 0.0) > (kc_previous or 0.0)

    liquidity_sweep_up = bool(getattr(market, "liquidity_sweep_up", False))
    liquidity_sweep_down = bool(getattr(market, "liquidity_sweep_down", False))

    bull_exhaustion = bool((range_pos or 50.0) >= 85 and (rsi or 50.0) >= 68 and momentum_falling) or liquidity_sweep_up
    bear_exhaustion = bool((range_pos or 50.0) <= 15 and (rsi or 50.0) <= 32 and momentum_rising) or liquidity_sweep_down

    warnings: list[str] = []
    if bull_exhaustion:
        warnings.append("Bull exhaustion risk: price is extended high or failed above resistance. Avoid chasing BUY.")
    if bear_exhaustion:
        warnings.append("Bear exhaustion risk: price is extended low or failed below support. Avoid chasing SELL.")
    if liquidity_sweep_up:
        warnings.append("Upside liquidity sweep detected.")
    if liquidity_sweep_down:
        warnings.append("Downside liquidity sweep detected.")

    if bull_exhaustion and bear_exhaustion:
        state = "two_way_exhaustion"
    elif bull_exhaustion:
        state = "bull_exhaustion"
    elif bear_exhaustion:
        state = "bear_exhaustion"
    else:
        state = "clear"

    return ExhaustionContext(
        state=state,
        block_buy=bool(bull_exhaustion),
        block_sell=bool(bear_exhaustion),
        bull_exhaustion=bool(bull_exhaustion),
        bear_exhaustion=bool(bear_exhaustion),
        liquidity_sweep_up=liquidity_sweep_up,
        liquidity_sweep_down=liquidity_sweep_down,
        rsi=rsi,
        kc_momentum=kc_momentum,
        range_position_pct=range_pos,
        warnings=warnings,
    )
