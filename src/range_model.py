from __future__ import annotations


def price_bands(market, regime: str) -> dict:
    price = float(market.price)
    atr = float(market.atr)
    bands = {
        "one_hour_low": price - 0.70 * atr,
        "one_hour_high": price + 0.70 * atr,
        "four_hour_low": price - 1.40 * atr,
        "four_hour_high": price + 1.40 * atr,
        "session_low": price - 1.80 * atr,
        "session_high": price + 1.80 * atr,
        "note": "ATR bands",
    }
    if regime == "bull_trend":
        bands["one_hour_high"] += 0.25 * atr
        bands["four_hour_high"] += 0.50 * atr
        bands["session_high"] += 0.75 * atr
        bands["note"] = "upper band extended"
    elif regime == "bear_trend":
        bands["one_hour_low"] -= 0.25 * atr
        bands["four_hour_low"] -= 0.50 * atr
        bands["session_low"] -= 0.75 * atr
        bands["note"] = "lower band extended"
    elif regime == "range":
        bands["four_hour_low"] = max(bands["four_hour_low"], float(market.support))
        bands["four_hour_high"] = min(bands["four_hour_high"], float(market.resistance))
        bands["session_low"] = max(bands["session_low"], float(market.support))
        bands["session_high"] = min(bands["session_high"], float(market.resistance))
        bands["note"] = "range boundary cap"
    elif regime == "shock":
        bands["note"] = "unstable band"
    return bands
