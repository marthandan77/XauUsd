from __future__ import annotations

import math

from .trade_plan import room_ratio


ACTIONABLE_PLANS = {"BUY PLAN", "SELL PLAN"}
UNCONFIRMED_RELEASE_REGIMES = {"compression", "squeeze_release_now", "squeeze_release_recent"}
OVEREXTENDED_RELEASE_REGIMES = {"bullish_release_overextended", "bearish_release_overextended"}
FILTER_HORIZONS = ["1h", "4h", "Daily"]


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


def _strongly_opposite(item: dict, action: str) -> bool:
    if item.get("status") != "ok":
        return False
    bias = item.get("bias")
    if action == "BUY PLAN":
        return bias == "bearish"
    if action == "SELL PLAN":
        return bias == "bullish"
    return False


def _add_horizon_alignment_veto(action: str, reasons: list[str], settings: dict) -> None:
    if action not in ACTIONABLE_PLANS:
        return
    if not bool(settings.get("require_horizon_alignment", True)):
        return

    horizons = _horizon_map()
    h5 = horizons.get("5m")
    if not h5 or h5.get("status") != "ok":
        reasons.append("5m signal forecast unavailable; cannot confirm scalp timing")
        return

    if action == "BUY PLAN":
        if h5.get("bias") != "bullish" or not _ev_ok(h5, "buy"):
            reasons.append("5m signal does not confirm buy edge above minimum EV")
    elif action == "SELL PLAN":
        if h5.get("bias") != "bearish" or not _ev_ok(h5, "sell"):
            reasons.append("5m signal does not confirm sell edge above minimum EV")

    for label in FILTER_HORIZONS:
        item = horizons.get(label)
        if not item:
            reasons.append(f"{label} filter unavailable")
            continue
        if item.get("status") != "ok":
            reasons.append(f"{label} filter not clean enough for final trade permission")
            continue
        if _strongly_opposite(item, action):
            reasons.append(f"{label} filter is opposite to the trade direction")


def apply_veto(action: str, plan: dict, market, regime: str, macro: dict, settings: dict, row: dict | None = None) -> dict:
    reasons: list[str] = []
    row = row or {}

    if bool(settings.get("signals_disabled", False)) and action in ACTIONABLE_PLANS:
        reasons.append(str(settings.get("signals_disabled_reason", "signals disabled")))
    if bool(row.get("data_stale", False)) and action in ACTIONABLE_PLANS:
        reasons.append("latest candle is stale; live scalp entry blocked")
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
