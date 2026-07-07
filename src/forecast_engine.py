from __future__ import annotations


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _blank_components() -> dict:
    return {
        "macro": {"bull": 0, "bear": 0},
        "trend": {"bull": 0, "bear": 0},
        "momentum": {"bull": 0, "bear": 0},
        "location": {"bull": 0, "bear": 0},
        "structure": {"bull": 0, "bear": 0},
        "volatility_risk": {"bull": 0, "bear": 0},
    }


def _add(components: dict, bucket: str, bull: int = 0, bear: int = 0) -> None:
    components[bucket]["bull"] += int(bull)
    components[bucket]["bear"] += int(bear)


def _totals(components: dict) -> tuple[int, int]:
    bull = sum(int(row.get("bull", 0)) for row in components.values())
    bear = sum(int(row.get("bear", 0)) for row in components.values())
    return bull, bear


def score_forecast(row: dict, market, regime: str, macro: dict, settings: dict) -> dict:
    components = _blank_components()
    notes: list[str] = []

    if macro.get("bias") in {"supportive", "mixed"}:
        _add(components, "macro", bull=20)
    if macro.get("bias") in {"restrictive", "mixed"}:
        _add(components, "macro", bear=20)
    if not macro.get("blocked"):
        _add(components, "volatility_risk", bull=10, bear=10)
    else:
        notes.append("Manual macro/news block is active.")

    if regime in {"bull_trend", "bullish_squeeze_breakout"}:
        _add(components, "trend", bull=25)
    elif regime in {"bear_trend", "bearish_squeeze_breakout"}:
        _add(components, "trend", bear=25)
    elif regime == "range":
        if getattr(market, "near_support", False):
            _add(components, "location", bull=8)
        if getattr(market, "near_resistance", False):
            _add(components, "location", bear=8)
    elif regime == "compression":
        notes.append("KC squeeze compression is active; wait for release direction.")
    elif regime == "shock":
        notes.append("Shock regime detected; directional forecast is lower quality.")

    if getattr(market, "near_support", False):
        _add(components, "location", bull=20)
    if getattr(market, "near_resistance", False):
        _add(components, "location", bear=20)
    if getattr(market, "near_vwap", False):
        _add(components, "location", bull=5, bear=5)
    if not getattr(market, "near_resistance", False):
        _add(components, "location", bull=5)
    if not getattr(market, "near_support", False):
        _add(components, "location", bear=5)

    if bool(getattr(market, "liquidity_sweep_down", False)):
        _add(components, "structure", bull=12)
        notes.append("Downside liquidity sweep detected; bearish continuation is lower quality.")
    if bool(getattr(market, "liquidity_sweep_up", False)):
        _add(components, "structure", bear=12)
        notes.append("Upside liquidity sweep detected; bullish continuation is lower quality.")

    close = _as_float(row.get("close"))
    open_ = _as_float(row.get("open", close), close)
    rsi = _as_float(row.get("rsi", 50), 50)
    level = _as_float(settings.get("rsi_level", 50), 50)
    if close > open_:
        _add(components, "momentum", bull=10)
    if close < open_:
        _add(components, "momentum", bear=10)
    if rsi >= level:
        _add(components, "momentum", bull=5)
    if rsi <= level:
        _add(components, "momentum", bear=5)

    kc_state = str(row.get("kc_state", "not_available"))
    kc_reason = str(row.get("kc_reason", ""))
    if bool(settings.get("kc_squeeze_enabled", True)):
        current_momentum = _as_float(row.get("kc_momentum"))
        previous_momentum = _as_float(row.get("kc_momentum_prev"))
        trend_sma = _as_float(row.get("trend_sma"), close)
        squeeze_fired = bool(row.get("squeeze_fired", False))
        squeeze_on = bool(row.get("squeeze_on", False))

        if squeeze_fired and current_momentum > 0 and current_momentum > previous_momentum and close > trend_sma:
            _add(components, "momentum", bull=int(settings.get("kc_breakout_score", 25)))
            kc_state = "bullish_squeeze_breakout"
            notes.append("KC squeeze fired upward with rising positive momentum.")
        elif squeeze_fired and current_momentum < 0 and current_momentum < previous_momentum and close < trend_sma:
            _add(components, "momentum", bear=int(settings.get("kc_breakout_score", 25)))
            kc_state = "bearish_squeeze_breakout"
            notes.append("KC squeeze fired downward with falling negative momentum.")
        elif squeeze_on:
            _add(components, "volatility_risk", bull=int(settings.get("kc_compression_score", 3)), bear=int(settings.get("kc_compression_score", 3)))
            kc_state = "compression"
            notes.append("KC compression detected; signal needs breakout confirmation.")
        elif current_momentum > 0 and close > trend_sma:
            _add(components, "momentum", bull=int(settings.get("kc_momentum_score", 10)))
            kc_state = "bullish_momentum"
        elif current_momentum < 0 and close < trend_sma:
            _add(components, "momentum", bear=int(settings.get("kc_momentum_score", 10)))
            kc_state = "bearish_momentum"

        if kc_reason:
            notes.append(kc_reason)

    _add(components, "structure", bull=5, bear=5)
    bull, bear = _totals(components)

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
        "components": components,
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
