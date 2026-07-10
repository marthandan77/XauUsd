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
    force_wait = False
    wait_reason = ""

    macro_bias = str(macro.get("bias", "mixed"))
    if macro.get("blocked"):
        force_wait = True
        wait_reason = "Manual event/news block is active; no fresh scalp entry."
        notes.append(wait_reason)
    else:
        notes.append(f"Macro context is {macro_bias}; used as context/filter, not as automatic score boost.")

    if regime in {"bull_trend", "bullish_release_confirmed"}:
        bull += 25
    elif regime in {"bear_trend", "bearish_release_confirmed"}:
        bear += 25
    elif regime == "range":
        bull += 8 if market.near_support else 0
        bear += 8 if market.near_resistance else 0
    elif regime == "compression":
        force_wait = True
        wait_reason = wait_reason or "KC squeeze compression is active; wait for release direction."
        notes.append(wait_reason)
    elif regime in {"squeeze_release_now", "squeeze_release_recent"}:
        force_wait = True
        wait_reason = wait_reason or "KC release is active but direction is not confirmed."
        notes.append(wait_reason)
    elif regime in {"bullish_release_overextended", "bearish_release_overextended"}:
        force_wait = True
        wait_reason = wait_reason or "KC release is confirmed but overextended; do not chase."
        notes.append(wait_reason)
    elif regime == "squeeze_release_expired":
        notes.append("Previous KC release window has expired; ignore old release.")

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
        squeeze_on = bool(row.get("squeeze_on", False))
        squeeze_fired = bool(row.get("squeeze_fired", False))
        squeeze_recent = bool(row.get("squeeze_recent", False))
        bullish_confirmed = bool(row.get("release_bullish_confirmed", False))
        bearish_confirmed = bool(row.get("release_bearish_confirmed", False))
        release_chase_risk = bool(row.get("release_chase_risk", False))

        if squeeze_on or kc_state == "compression":
            kc_state = "compression"
            force_wait = True
            wait_reason = wait_reason or "KC compression detected; signal needs breakout confirmation."
            notes.append("KC compression detected; signal needs breakout confirmation.")
        elif release_chase_risk or kc_state in {"bullish_release_overextended", "bearish_release_overextended"}:
            force_wait = True
            wait_reason = wait_reason or "KC release is overextended; do not chase."
            notes.append("KC release is overextended; do not chase.")
        elif squeeze_recent and bullish_confirmed:
            bull += int(settings.get("kc_breakout_score", 25))
            kc_state = "bullish_release_confirmed"
            notes.append("KC release confirmed bullish: price broke release high with positive/rising momentum.")
            if close <= trend_sma:
                notes.append("Bullish release is lower quality because price is not above the trend SMA.")
        elif squeeze_recent and bearish_confirmed:
            bear += int(settings.get("kc_breakout_score", 25))
            kc_state = "bearish_release_confirmed"
            notes.append("KC release confirmed bearish: price broke release low with negative/falling momentum.")
            if close >= trend_sma:
                notes.append("Bearish release is lower quality because price is not below the trend SMA.")
        elif squeeze_fired:
            kc_state = "release_fired_now"
            force_wait = True
            wait_reason = wait_reason or "KC release fired on this candle; wait for directional confirmation."
            notes.append("KC release fired on this candle; wait for directional confirmation.")
        elif squeeze_recent:
            kc_state = "release_recent_unconfirmed"
            force_wait = True
            wait_reason = wait_reason or "KC release is recent but unconfirmed."
            notes.append("KC release is recent but unconfirmed.")
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

    if bool(settings.get("signals_disabled", False)):
        force_wait = False
        wait_reason = str(settings.get("signals_disabled_reason", "Signals disabled."))
        notes.append(wait_reason)

    if bull > bear:
        bias = "bullish"
        confidence = bull
    elif bear > bull:
        bias = "bearish"
        confidence = bear
    else:
        bias = "mixed"
        confidence = bull

    rule_score = int(min(max(confidence, 0), 100))
    return {
        "bull_score": int(bull),
        "bear_score": int(bear),
        "bias": bias,
        "confidence": rule_score,
        "rule_score": rule_score,
        "kc_state": kc_state,
        "kc_reason": kc_reason,
        "force_wait": bool(force_wait),
        "wait_reason": wait_reason,
        "notes": notes,
    }


def choose_action(scores: dict, settings: dict) -> str:
    if bool(settings.get("signals_disabled", False)):
        return "HOLD"

    if bool(scores.get("force_wait", False)):
        return "WAIT"

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
