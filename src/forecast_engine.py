from __future__ import annotations


def score_forecast(row: dict, market, regime: str, macro: dict, settings: dict) -> dict:
    bull = 0
    bear = 0
    if macro.get("bias") in {"supportive", "mixed"}:
        bull += 20
    if macro.get("bias") in {"restrictive", "mixed"}:
        bear += 20
    if not macro.get("blocked"):
        bull += 10
        bear += 10
    if regime == "bull_trend":
        bull += 25
    elif regime == "bear_trend":
        bear += 25
    elif regime == "range":
        bull += 8 if market.near_support else 0
        bear += 8 if market.near_resistance else 0
    if market.near_support:
        bull += 20
    if market.near_resistance:
        bear += 20
    if market.near_vwap:
        bull += 5
        bear += 5
    if not market.near_resistance:
        bull += 5
    if not market.near_support:
        bear += 5
    close = float(row.get("close", 0))
    open_ = float(row.get("open", close))
    rsi = float(row.get("rsi", 50))
    level = float(settings.get("rsi_level", 50))
    if close > open_:
        bull += 10
    if close < open_:
        bear += 10
    if rsi >= level:
        bull += 5
    if rsi <= level:
        bear += 5
    bull += 5
    bear += 5
    if bull > bear:
        bias = "bullish"
        confidence = bull
    elif bear > bull:
        bias = "bearish"
        confidence = bear
    else:
        bias = "mixed"
        confidence = bull
    return {"bull_score": int(bull), "bear_score": int(bear), "bias": bias, "confidence": int(confidence)}


def choose_action(scores: dict, settings: dict) -> str:
    bull = scores["bull_score"]
    bear = scores["bear_score"]
    if bull >= int(settings.get("buy_threshold", 78)) and bull > bear:
        return "BUY PLAN"
    if bear >= int(settings.get("sell_threshold", 78)) and bear > bull:
        return "SELL PLAN"
    if max(bull, bear) >= int(settings.get("wait_threshold", 60)):
        return "WAIT"
    return "HOLD"
