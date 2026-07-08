from __future__ import annotations

import math

import pandas as pd


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _as_float(value, default: float | None = None):
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _summary(
    *,
    state: str,
    bias: str,
    action_hint: str,
    confidence_boost: int,
    reason: str,
    row: pd.Series,
    price=None,
    trend_sma=None,
    current_momentum=None,
    previous_momentum=None,
) -> dict:
    return {
        "state": state,
        "bias": bias,
        "action_hint": action_hint,
        "confidence_boost": int(confidence_boost),
        "reason": reason,
        "squeeze_on": bool(row.get("squeeze_on", False)),
        "squeeze_fired": bool(row.get("squeeze_fired", False)),
        "squeeze_recent": bool(row.get("squeeze_recent", False)),
        "bars_since_squeeze_release": _as_float(row.get("bars_since_squeeze_release")),
        "compression_ratio": _as_float(row.get("compression_ratio")),
        "release_direction": str(row.get("release_direction", "none")),
        "release_close": _as_float(row.get("release_close")),
        "release_high": _as_float(row.get("release_high")),
        "release_low": _as_float(row.get("release_low")),
        "release_chase_atr": _as_float(row.get("release_chase_atr")),
        "release_chase_risk": bool(row.get("release_chase_risk", False)),
        "release_bullish_confirmed": bool(row.get("release_bullish_confirmed", False)),
        "release_bearish_confirmed": bool(row.get("release_bearish_confirmed", False)),
        "current_momentum": _as_float(current_momentum if current_momentum is not None else row.get("kc_momentum")),
        "previous_momentum": _as_float(previous_momentum if previous_momentum is not None else row.get("kc_momentum_prev")),
        "trend_sma": _as_float(trend_sma if trend_sma is not None else row.get("trend_sma")),
        "price": _as_float(price if price is not None else row.get("close")),
    }


def kc_squeeze_summary(df: pd.DataFrame, settings: dict) -> dict:
    """Return a KC Squeeze state-machine summary from already-enriched OHLCV data.

    Release detection is separated from trade qualification. A raw release is context;
    confirmed direction plus clean veto filters are required before a trade plan.
    """
    empty_row = pd.Series(dtype="object")
    if df is None or df.empty or len(df) < 3:
        return _summary(
            state="insufficient_data",
            bias="neutral",
            action_hint="NO TRADE",
            confidence_boost=0,
            reason="Not enough data for KC Squeeze calculation.",
            row=empty_row,
        )

    row = df.iloc[-1]
    price = _as_float(row.get("close"))
    trend_sma = _as_float(row.get("trend_sma"))
    current_momentum = _as_float(row.get("kc_momentum"))
    previous_momentum = _as_float(row.get("kc_momentum_prev"))

    if not (_finite(price) and _finite(trend_sma) and _finite(current_momentum) and _finite(previous_momentum)):
        return _summary(
            state="insufficient_data",
            bias="neutral",
            action_hint="NO TRADE",
            confidence_boost=0,
            reason="KC Squeeze indicators are still warming up.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    squeeze_on = bool(row.get("squeeze_on", False))
    squeeze_fired = bool(row.get("squeeze_fired", False))
    squeeze_recent = bool(row.get("squeeze_recent", False))
    release_bullish = bool(row.get("release_bullish_confirmed", False))
    release_bearish = bool(row.get("release_bearish_confirmed", False))
    release_chase_risk = bool(row.get("release_chase_risk", False))
    bars_since = _as_float(row.get("bars_since_squeeze_release"))
    release_chase_atr = _as_float(row.get("release_chase_atr"))
    chase_limit = float(settings.get("kc_release_chase_atr_limit", 1.0))

    above_trend = float(price) > float(trend_sma)
    below_trend = float(price) < float(trend_sma)
    momentum_rising = float(current_momentum) > float(previous_momentum)
    momentum_falling = float(current_momentum) < float(previous_momentum)

    if squeeze_on:
        return _summary(
            state="compression",
            bias="neutral",
            action_hint="WAIT",
            confidence_boost=5,
            reason="Bollinger Bands are inside the Keltner Channel. Volatility is compressed; wait for release direction.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    if squeeze_recent and release_bullish:
        if release_chase_risk:
            return _summary(
                state="bullish_release_overextended",
                bias="bullish",
                action_hint="WAIT / DO NOT CHASE",
                confidence_boost=8,
                reason=f"Bullish KC release is confirmed, but price has moved {release_chase_atr:.2f} ATR from the release close; chase limit is {chase_limit:.2f} ATR.",
                row=row,
                price=price,
                trend_sma=trend_sma,
                current_momentum=current_momentum,
                previous_momentum=previous_momentum,
            )
        quality_note = "above the trend SMA" if above_trend else "not above the trend SMA"
        return _summary(
            state="bullish_release_confirmed",
            bias="bullish",
            action_hint="BUY PLAN CANDIDATE",
            confidence_boost=25,
            reason=f"KC release is bullish: price broke above the release high, momentum is positive/rising, and price is {quality_note}.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    if squeeze_recent and release_bearish:
        if release_chase_risk:
            return _summary(
                state="bearish_release_overextended",
                bias="bearish",
                action_hint="WAIT / DO NOT CHASE",
                confidence_boost=8,
                reason=f"Bearish KC release is confirmed, but price has moved {release_chase_atr:.2f} ATR from the release close; chase limit is {chase_limit:.2f} ATR.",
                row=row,
                price=price,
                trend_sma=trend_sma,
                current_momentum=current_momentum,
                previous_momentum=previous_momentum,
            )
        quality_note = "below the trend SMA" if below_trend else "not below the trend SMA"
        return _summary(
            state="bearish_release_confirmed",
            bias="bearish",
            action_hint="SELL PLAN CANDIDATE",
            confidence_boost=25,
            reason=f"KC release is bearish: price broke below the release low, momentum is negative/falling, and price is {quality_note}.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    if squeeze_fired:
        return _summary(
            state="release_fired_now",
            bias="neutral",
            action_hint="WAIT",
            confidence_boost=8,
            reason="KC compression has just released on the latest candle. Direction is not confirmed yet.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    if squeeze_recent:
        return _summary(
            state="release_recent_unconfirmed",
            bias="neutral",
            action_hint="WAIT",
            confidence_boost=6,
            reason=f"KC release occurred {bars_since:.0f} bars ago, but direction is not confirmed yet.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    if bars_since is not None:
        if current_momentum > 0 and above_trend and momentum_rising:
            return _summary(
                state="bullish_momentum",
                bias="bullish",
                action_hint="HOLD / WATCH",
                confidence_boost=10,
                reason="Momentum is positive/rising and price is above the trend SMA, but there is no fresh KC release.",
                row=row,
                price=price,
                trend_sma=trend_sma,
                current_momentum=current_momentum,
                previous_momentum=previous_momentum,
            )
        if current_momentum < 0 and below_trend and momentum_falling:
            return _summary(
                state="bearish_momentum",
                bias="bearish",
                action_hint="EXIT LONG / AVOID BUY",
                confidence_boost=10,
                reason="Momentum is negative/falling and price is below the trend SMA. Long entries are not favored.",
                row=row,
                price=price,
                trend_sma=trend_sma,
                current_momentum=current_momentum,
                previous_momentum=previous_momentum,
            )
        return _summary(
            state="release_expired",
            bias="neutral",
            action_hint="HOLD",
            confidence_boost=0,
            reason="The previous KC release window has expired. Do not chase the old release.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    if current_momentum > 0 and above_trend:
        return _summary(
            state="bullish_momentum",
            bias="bullish",
            action_hint="HOLD / WATCH",
            confidence_boost=10,
            reason="Momentum is positive and price is above the trend SMA, but there is no fresh squeeze breakout.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    if current_momentum < 0 and below_trend:
        return _summary(
            state="bearish_momentum",
            bias="bearish",
            action_hint="EXIT LONG / AVOID BUY",
            confidence_boost=10,
            reason="Momentum is negative and price is below the trend SMA. Long entries are not favored.",
            row=row,
            price=price,
            trend_sma=trend_sma,
            current_momentum=current_momentum,
            previous_momentum=previous_momentum,
        )

    return _summary(
        state="mixed",
        bias="neutral",
        action_hint="HOLD",
        confidence_boost=0,
        reason="KC Squeeze conditions are mixed. No clean edge.",
        row=row,
        price=price,
        trend_sma=trend_sma,
        current_momentum=current_momentum,
        previous_momentum=previous_momentum,
    )
