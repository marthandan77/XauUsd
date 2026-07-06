from __future__ import annotations


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def score_forecast(row: dict, market, regime: str, macro: dict, settings: dict) -> dict:
    bull = 0
    bear = 0
    notes: list[str] = []

    if macro.get("bias") in {"supportive", "mixed"}:
        bull += 20
    if macro.get("bias") in {"restrictive", "mixed"}:
        bear += 20
    if not macro.get("blocked"):
        bull += 10
        bear += 10

    if regime in {"bull_trend", "bullish_squeeze_breakout"}:
        bull += 25
    elif regime in {"bear_trend", "bearish_squeeze_breakout"}:
        bear += 25
    elif regime == "range":
        bull += 8 if market.near_support else 0
        bear += 8 if market.near_resistance else 0
    elif regime == "compression":
        notes.append("KC squeeze compression is active; wait for release direction.")

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

    close = _as_float(row.get("close"))
    open_ = _as_float(row.get("open", close), close)
    rsi = _as_float(row.get("rsi", 50), 50)
    level = _as_float(settings.get("rsi_level", 50), 50)
    if close > open_:
        bull += 10
    if close < open_:
        bear += 10
    if rsi >= level:
        bull += 5
    if rsi <= level:
        bear += 5

    kc_state = str(row.get("kc_state", "not_available"))
    kc_reason = str(row.get("kc_reason", ""))
    if bool(settings.get("kc_squeeze_enabled", True)):
        current_momentum = _as_float(row.get("kc_momentum"))
        previous_momentum = _as_float(row.get("kc_momentum_prev"))
        trend_sma = _as_float(row.get("trend_sma"), close)
        squeeze_fired = bool(row.get("squeeze_fired", False))
        squeeze_on = bool(row.get("squeeze_on", False))

        if squeeze_fired and current_momentum > 0 and current_momentum > previous_momentum and close > trend_sma:
            bull += int(settings.get("kc_breakout_score", 25))
            kc_state = "bullish_squeeze_breakout"
            notes.append("KC squeeze fired upward with rising positive momentum.")
        elif squeeze_fired and current_momentum < 0 and current_momentum < previous_momentum and close < trend_sma:
            bear += int(settings.get("kc_breakout_score", 25))
            kc_state = "bearish_squeeze_breakout"
            notes.append("KC squeeze fired downward with falling negative momentum.")
        elif squeeze_on:
            bull += int(settings.get("kc_compression_score", 3))
            bear += int(settings.get("kc_compression_score", 3))
            kc_state = "compression"
            notes.append("KC compression detected; signal needs breakout confirmation.")
        elif current_momentum > 0 and close > trend_sma:
            bull += int(settings.get("kc_momentum_score", 10))
            kc_state = "bullish_momentum"
        elif current_momentum < 0 and close < trend_sma:
            bear += int(settings.get("kc_momentum_score", 10))
            kc_state = "bearish_momentum"

        if kc_reason:
            notes.append(kc_reason)

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

    return {
        "bull_score": int(bull),
        "bear_score": int(bear),
        "bias": bias,
        "confidence": int(min(max(confidence, 0), 100)),
        "kc_state": kc_state,
        "kc_reason": kc_reason,
        "notes": notes,
    }


def choose_action(scores: dict, settings: dict) -> str:
    bull = scores["bull_score"]
    bear = scores["bear_score"]
    long_enabled = bool(settings.get("long_plans_enabled", True))
    short_enabled = bool(settings.get("short_plans_enabled", False))

    if long_enabled and bull >= int(settings.get("buy_threshold", 78)) and bull > bear:
        return "BUY PLAN"
    if bear >= int(settings.get("sell_threshold", 78)) and bear > bull:
        if short_enabled:
            return "SELL PLAN"
        return "EXIT LONG / AVOID BUY"
    if max(bull, bear) >= int(settings.get("wait_threshold", 60)):
        return "WAIT"
    return "HOLD"
