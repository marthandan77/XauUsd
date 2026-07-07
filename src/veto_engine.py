from __future__ import annotations

from .trade_plan import room_ratio

ACTIONABLE_PLANS = {"BUY PLAN", "SELL PLAN"}


def apply_veto(action: str, plan: dict, market, regime: str, macro: dict, settings: dict, exhaustion: dict | None = None) -> dict:
    reasons: list[str] = []
    if macro.get("blocked"):
        reasons.append("event block active")
    if regime == "shock":
        reasons.append("shock regime")
    if regime == "compression" and action in ACTIONABLE_PLANS:
        reasons.append("squeeze still compressed; wait for release")
    if market.middle_range and action in ACTIONABLE_PLANS:
        reasons.append("middle of range")
    if action == "BUY PLAN" and market.near_resistance:
        reasons.append("too close to resistance")
    if action == "SELL PLAN" and market.near_support:
        reasons.append("too close to support")
    if action == "SELL PLAN" and not bool(settings.get("short_plans_enabled", False)):
        reasons.append("short plans disabled")

    if exhaustion:
        if action == "BUY PLAN" and bool(exhaustion.get("block_buy", False)):
            reasons.append("bull exhaustion risk; avoid chasing BUY")
        if action == "SELL PLAN" and bool(exhaustion.get("block_sell", False)):
            reasons.append("bear exhaustion risk; avoid chasing SELL")

    rr = room_ratio(action, plan, market)
    if action in ACTIONABLE_PLANS and rr < float(settings.get("min_reward_risk", 1.5)):
        reasons.append("not enough room to target")

    risk = float(plan.get("risk", 0) or 0)
    atr = float(market.atr)
    if action in ACTIONABLE_PLANS:
        if risk < float(settings.get("min_sl_atr_fraction", 0.5)) * atr:
            reasons.append("stop too tight")
        if risk > float(settings.get("max_sl_atr_fraction", 3.5)) * atr:
            reasons.append("stop too wide")

    if reasons and action in ACTIONABLE_PLANS:
        return {"final_action": "HOLD", "trade_quality": "Rejected", "reasons": reasons, "room_ratio": rr}

    if action == "BUY PLAN" and not reasons:
        quality = "Clean"
    elif action == "SELL PLAN" and not reasons:
        quality = "Clean"
    elif action in {"WAIT", "EXIT LONG / AVOID BUY"}:
        quality = "Watch"
    else:
        quality = "No edge"

    return {"final_action": action, "trade_quality": quality, "reasons": reasons, "room_ratio": rr}
