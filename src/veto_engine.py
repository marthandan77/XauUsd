from __future__ import annotations

from .trade_plan import room_ratio

ACTIONABLE_PLANS = {"BUY PLAN", "SELL PLAN"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number


def apply_veto(action: str, plan: dict, market, regime: str, macro: dict, settings: dict | None = None, exhaustion: dict | None = None, *_, **__) -> dict:
    """Apply final safety vetoes.

    The loose trailing args keep Streamlit Cloud safe during redeploys where app.py and
    src/veto_engine.py can briefly be out of sync.
    """
    settings = settings or {}
    macro = macro or {}
    plan = plan or {}
    reasons: list[str] = []

    if macro.get("blocked"):
        reasons.append("event block active")
    if regime == "shock":
        reasons.append("shock regime")
    if regime == "compression" and action in ACTIONABLE_PLANS:
        reasons.append("squeeze still compressed; wait for release")
    if bool(getattr(market, "middle_range", False)) and action in ACTIONABLE_PLANS:
        reasons.append("middle of range")
    if action == "BUY PLAN" and bool(getattr(market, "near_resistance", False)):
        reasons.append("too close to resistance")
    if action == "SELL PLAN" and bool(getattr(market, "near_support", False)):
        reasons.append("too close to support")
    if action == "SELL PLAN" and not bool(settings.get("short_plans_enabled", False)):
        reasons.append("short plans disabled")

    if isinstance(exhaustion, dict):
        if action == "BUY PLAN" and bool(exhaustion.get("block_buy", False)):
            reasons.append("bull exhaustion risk; avoid chasing BUY")
        if action == "SELL PLAN" and bool(exhaustion.get("block_sell", False)):
            reasons.append("bear exhaustion risk; avoid chasing SELL")

    try:
        rr = room_ratio(action, plan, market)
    except Exception:
        rr = 0.0

    if action in ACTIONABLE_PLANS and rr < _safe_float(settings.get("min_reward_risk", 1.5), 1.5):
        reasons.append("not enough room to target")

    risk = _safe_float(plan.get("risk", 0), 0.0)
    atr = _safe_float(getattr(market, "atr", 0), 0.0)
    if action in ACTIONABLE_PLANS and atr > 0:
        if risk < _safe_float(settings.get("min_sl_atr_fraction", 0.5), 0.5) * atr:
            reasons.append("stop too tight")
        if risk > _safe_float(settings.get("max_sl_atr_fraction", 3.5), 3.5) * atr:
            reasons.append("stop too wide")

    if reasons and action in ACTIONABLE_PLANS:
        return {"final_action": "HOLD", "trade_quality": "Rejected", "reasons": reasons, "room_ratio": rr}

    if action in {"BUY PLAN", "SELL PLAN"} and not reasons:
        quality = "Clean"
    elif action in {"WAIT", "EXIT LONG / AVOID BUY"}:
        quality = "Watch"
    else:
        quality = "No edge"

    return {"final_action": action, "trade_quality": quality, "reasons": reasons, "room_ratio": rr}
