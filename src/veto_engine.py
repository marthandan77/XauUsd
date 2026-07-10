from __future__ import annotations

import math

from .trade_plan import room_ratio


ACTIONABLE_PLANS = {"BUY PLAN", "SELL PLAN"}
UNCONFIRMED_RELEASE_REGIMES = {"compression", "squeeze_release_now", "squeeze_release_recent"}
OVEREXTENDED_RELEASE_REGIMES = {"bullish_release_overextended", "bearish_release_overextended"}


def _finite_float(value, default: float | None = None):
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def _horizon_map() -> dict[str, dict]:
    try:
        import streamlit as st
    except Exception:
        return {}
    forecasts = st.session_state.get("_xauusd_multi_horizon_forecasts") or []
    return {str(item.get("horizon")): item for item in forecasts if isinstance(item, dict)}


def _ev_ok(item: dict, side: str) -> bool:
    key = "ev_buy_points" if side == "buy" else "ev_sell_points"
    ev = _finite_float(item.get(key))
    min_ev = _finite_float(item.get("min_ev_points"), 0.0)
    if ev is None:
        return False
    return ev >= float(min_ev or 0.0)


def _add_horizon_alignment_veto(action: str, reasons: list[str], settings: dict) -> None:
    if action not in ACTIONABLE_PLANS:
        return
    if not bool(settings.get("require_horizon_alignment", True)):
        return

    horizons = _horizon_map()
    h15 = horizons.get("15m")
    h30 = horizons.get("30m")
    h1h = horizons.get("1h")

    if not h15 or not h30:
        reasons.append("multi-horizon forecast unavailable; cannot confirm 15m/30m agreement")
        return
    if h15.get("status") != "ok" or h30.get("status") != "ok":
        reasons.append("multi-horizon forecast not clean enough for final trade permission")
        return

    if action == "BUY PLAN":
        if h15.get("bias") != "bullish" or h30.get("bias") != "bullish":
            reasons.append("15m and 30m predictors do not agree bullish")
        if not _ev_ok(h15, "buy") or not _ev_ok(h30, "buy"):
            reasons.append("15m/30m buy EV is below minimum threshold")
        if h1h and h1h.get("status") == "ok" and h1h.get("bias") == "bearish":
            reasons.append("1h predictor is bearish danger filter")

    if action == "SELL PLAN":
        if h15.get("bias") != "bearish" or h30.get("bias") != "bearish":
            reasons.append("15m and 30m predictors do not agree bearish")
        if not _ev_ok(h15, "sell") or not _ev_ok(h30, "sell"):
            reasons.append("15m/30m sell EV is below minimum threshold")
        if h1h and h1h.get("status") == "ok" and h1h.get("bias") == "bullish":
            reasons.append("1h predictor is bullish danger filter")


def apply_veto(action: str, plan: dict, market, regime: str, macro: dict, settings: dict, row: dict | None = None) -> dict:
    reasons: list[str] = []
    row = row or {}

    if bool(settings.get("signals_disabled", False)) and action in ACTIONABLE_PLANS:
        reasons.append(str(settings.get("signals_disabled_reason", "signals disabled")))
    if macro.get("blocked"):
        reasons.append("event block active")
    if regime == "shock":
        reasons.append("shock regime")
    if regime == "compression" and action in ACTIONABLE_PLANS:
        reasons.append("squeeze still compressed; wait for release")
    if regime in UNCONFIRMED_RELEASE_REGIMES and action in ACTIONABLE_PLANS:
        reasons.append("KC release direction unconfirmed")
    if regime in OVEREXTENDED_RELEASE_REGIMES and action in ACTIONABLE_PLANS:
        reasons.append("KC release overextended; do not chase")

    release_chase_atr = _finite_float(row.get("release_chase_atr"))
    if action in ACTIONABLE_PLANS and release_chase_atr is not None:
        chase_limit = float(settings.get("kc_release_chase_atr_limit", 1.0))
        if release_chase_atr > chase_limit:
            reasons.append(f"KC release chase risk above limit: {release_chase_atr:.2f} ATR > {chase_limit:.2f} ATR")

    if action in ACTIONABLE_PLANS and bool(plan.get("tp1_too_close", False)):
        tp1_distance = _finite_float(plan.get("tp1_distance_atr"), 0.0)
        min_tp1_distance = float(settings.get("min_tp1_atr_distance", 0.9))
        reasons.append(f"TP1 too close to entry: {tp1_distance:.2f} ATR < {min_tp1_distance:.2f} ATR minimum")

    if market.middle_range and action in ACTIONABLE_PLANS:
        reasons.append("middle of range")
    if action == "BUY PLAN" and market.near_resistance:
        reasons.append("too close to resistance")
    if action == "SELL PLAN" and market.near_support:
        reasons.append("too close to support")
    if action == "SELL PLAN" and not bool(settings.get("short_plans_enabled", False)):
        reasons.append("short plans disabled")

    rr = room_ratio(action, plan, market)
    if action in ACTIONABLE_PLANS and rr < float(settings.get("min_reward_risk", 1.2)):
        reasons.append("not enough room to target")

    risk = float(plan.get("risk", 0) or 0)
    atr = float(market.atr)
    if action in ACTIONABLE_PLANS:
        if risk < float(settings.get("min_sl_atr_fraction", 0.5)) * atr:
            reasons.append("stop too tight")
        if risk > float(settings.get("max_sl_atr_fraction", 3.5)) * atr:
            reasons.append("stop too wide")

    _add_horizon_alignment_veto(action, reasons, settings)

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
