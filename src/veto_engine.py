from __future__ import annotations

from .trade_plan import room_ratio


def apply_veto(action: str, plan: dict, market, regime: str, macro: dict, settings: dict) -> dict:
    reasons: list[str] = []
    if macro.get("blocked"):
        reasons.append("event block active")
    if regime == "shock":
        reasons.append("shock regime")
    if market.middle_range:
        reasons.append("middle of range")
    if action == "BUY PLAN" and market.near_resistance:
        reasons.append("too close to resistance")
    if action == "SELL PLAN" and market.near_support:
        reasons.append("too close to support")
    rr = room_ratio(action, plan, market)
    if action in {"BUY PLAN", "SELL PLAN"} and rr < float(settings.get("min_reward_risk", 1.5)):
        reasons.append("not enough room to target")
    risk = float(plan.get("risk", 0) or 0)
    atr = float(market.atr)
    if action in {"BUY PLAN", "SELL PLAN"}:
        if risk < float(settings.get("min_sl_atr_fraction", 0.5)) * atr:
            reasons.append("stop too tight")
        if risk > float(settings.get("max_sl_atr_fraction", 2.5)) * atr:
            reasons.append("stop too wide")
    if reasons and action in {"BUY PLAN", "SELL PLAN"}:
        return {"final_action": "HOLD", "trade_quality": "Rejected", "reasons": reasons, "room_ratio": rr}
    quality = "Clean" if action in {"BUY PLAN", "SELL PLAN"} and not reasons else "Watch" if action == "WAIT" else "No edge"
    return {"final_action": action, "trade_quality": quality, "reasons": reasons, "room_ratio": rr}
