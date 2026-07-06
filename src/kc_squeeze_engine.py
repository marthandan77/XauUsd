from __future__ import annotations

import math

import pandas as pd


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def kc_squeeze_summary(df: pd.DataFrame, settings: dict) -> dict:
    """Return an advisory KC Squeeze state from already-enriched OHLCV data.

    This module converts the QuantConnect KC Squeeze idea into dashboard logic.
    It does not place orders and does not assume a position exists.
    """
    if df is None or df.empty or len(df) < 3:
        return {
            "state": "insufficient_data",
            "bias": "neutral",
            "action_hint": "NO TRADE",
            "confidence_boost": 0,
            "reason": "Not enough data for KC Squeeze calculation.",
            "squeeze_on": False,
            "squeeze_fired": False,
            "current_momentum": None,
            "previous_momentum": None,
            "trend_sma": None,
        }

    row = df.iloc[-1]
    price = float(row.get("close", 0) or 0)
    trend_sma = row.get("trend_sma")
    current_momentum = row.get("kc_momentum")
    previous_momentum = row.get("kc_momentum_prev")
    squeeze_on = bool(row.get("squeeze_on", False))
    squeeze_fired = bool(row.get("squeeze_fired", False))

    if not (_finite(price) and _finite(trend_sma) and _finite(current_momentum) and _finite(previous_momentum)):
        return {
            "state": "insufficient_data",
            "bias": "neutral",
            "action_hint": "NO TRADE",
            "confidence_boost": 0,
            "reason": "KC Squeeze indicators are still warming up.",
            "squeeze_on": squeeze_on,
            "squeeze_fired": squeeze_fired,
            "current_momentum": None if not _finite(current_momentum) else float(current_momentum),
            "previous_momentum": None if not _finite(previous_momentum) else float(previous_momentum),
            "trend_sma": None if not _finite(trend_sma) else float(trend_sma),
        }

    current_momentum = float(current_momentum)
    previous_momentum = float(previous_momentum)
    trend_sma = float(trend_sma)
    momentum_rising = current_momentum > previous_momentum
    momentum_falling = current_momentum < previous_momentum
    above_trend = price > trend_sma
    below_trend = price < trend_sma

    if squeeze_fired and current_momentum > 0 and momentum_rising and above_trend:
        return {
            "state": "bullish_squeeze_breakout",
            "bias": "bullish",
            "action_hint": "BUY PLAN",
            "confidence_boost": 25,
            "reason": "Squeeze fired upward: momentum is positive, rising, and price is above the trend SMA.",
            "squeeze_on": squeeze_on,
            "squeeze_fired": squeeze_fired,
            "current_momentum": current_momentum,
            "previous_momentum": previous_momentum,
            "trend_sma": trend_sma,
        }

    if squeeze_fired and current_momentum < 0 and momentum_falling and below_trend:
        return {
            "state": "bearish_squeeze_breakout",
            "bias": "bearish",
            "action_hint": "EXIT LONG / AVOID BUY",
            "confidence_boost": 25,
            "reason": "Squeeze fired downward: momentum is negative, falling, and price is below the trend SMA.",
            "squeeze_on": squeeze_on,
            "squeeze_fired": squeeze_fired,
            "current_momentum": current_momentum,
            "previous_momentum": previous_momentum,
            "trend_sma": trend_sma,
        }

    if squeeze_on:
        return {
            "state": "compression",
            "bias": "neutral",
            "action_hint": "WAIT",
            "confidence_boost": 5,
            "reason": "Bollinger Bands are inside the Keltner Channel. Volatility is compressed; wait for release direction.",
            "squeeze_on": squeeze_on,
            "squeeze_fired": squeeze_fired,
            "current_momentum": current_momentum,
            "previous_momentum": previous_momentum,
            "trend_sma": trend_sma,
        }

    if current_momentum > 0 and above_trend:
        return {
            "state": "bullish_momentum",
            "bias": "bullish",
            "action_hint": "HOLD / WATCH",
            "confidence_boost": 10,
            "reason": "Momentum is positive and price is above the trend SMA, but there is no fresh squeeze breakout.",
            "squeeze_on": squeeze_on,
            "squeeze_fired": squeeze_fired,
            "current_momentum": current_momentum,
            "previous_momentum": previous_momentum,
            "trend_sma": trend_sma,
        }

    if current_momentum < 0 and below_trend:
        return {
            "state": "bearish_momentum",
            "bias": "bearish",
            "action_hint": "EXIT LONG / AVOID BUY",
            "confidence_boost": 10,
            "reason": "Momentum is negative and price is below the trend SMA. Long entries are not favored.",
            "squeeze_on": squeeze_on,
            "squeeze_fired": squeeze_fired,
            "current_momentum": current_momentum,
            "previous_momentum": previous_momentum,
            "trend_sma": trend_sma,
        }

    return {
        "state": "mixed",
        "bias": "neutral",
        "action_hint": "HOLD",
        "confidence_boost": 0,
        "reason": "KC Squeeze conditions are mixed. No clean edge.",
        "squeeze_on": squeeze_on,
        "squeeze_fired": squeeze_fired,
        "current_momentum": current_momentum,
        "previous_momentum": previous_momentum,
        "trend_sma": trend_sma,
    }
