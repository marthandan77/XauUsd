from __future__ import annotations

import math


def _floor_below_price(price: float, target: float, minimum_gap: float) -> float:
    """Keep a short-side target below current price."""
    if target >= price:
        return price - minimum_gap
    return target


def _sell_target_before_support(price: float, raw_target: float, support_level: float, atr: float, buffer_atr: float) -> float:
    """For shorts, avoid aiming far below nearby support. Exit before the level instead."""
    if support_level < price:
        return max(raw_target, support_level + buffer_atr * atr)
    return raw_target


def _tp1_distance_atr(side: str, entry_low: float, entry_high: float, tp1: float, atr: float) -> float:
    """Measure clean distance from the realistic entry edge to TP1 in ATR units."""
    if atr <= 0:
        return 0.0
    if side == "buy":
        return max((tp1 - entry_high) / atr, 0.0)
    if side == "sell":
        return max((entry_low - tp1) / atr, 0.0)
    return 0.0


def _tp1_too_close(distance_atr: float, settings: dict) -> bool:
    return distance_atr < float(settings.get("min_tp1_atr_distance", 0.9))


def build_trade_plan(action: str, market, settings: dict) -> dict:
    price = float(market.price)
    atr = float(market.atr)
    stop_mult = float(settings.get("atr_stop_multiplier", settings.get("atr_multiplier", 1.6)))
    min_tp1_distance = float(settings.get("min_tp1_atr_distance", 0.9))
    account_size = float(settings.get("account_size", 10000))
    risk_pct = float(settings.get("risk_per_trade_pct", 0.5)) / 100.0

    if action == "BUY PLAN":
        stop = min(float(market.swing_low), price - stop_mult * atr)
        risk = max(price - stop, 0.01)
        buy_tp1_mult = float(settings.get("buy_tp1_atr_multiplier", 1.2))
        buy_tp2_mult = float(settings.get("buy_tp2_atr_multiplier", 2.0))
        entry_low = price - 0.20 * atr
        entry_high = price + 0.10 * atr
        tp1 = price + buy_tp1_mult * atr
        tp2 = price + max(buy_tp2_mult, buy_tp1_mult + 0.50) * atr
        tp1_distance_atr = _tp1_distance_atr("buy", entry_low, entry_high, tp1, atr)
        tp1_too_close = _tp1_too_close(tp1_distance_atr, settings)
        quantity = advisory_position_size(account_size, risk_pct, (entry_low + entry_high) / 2.0, stop)
        return {
            "side": "buy_advisory",
            "quantity": quantity,
            "entry_zone_low": entry_low,
            "entry_zone_high": entry_high,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "risk": risk,
            "tp1_distance_atr": tp1_distance_atr,
            "tp1_too_close": tp1_too_close,
            "min_tp1_atr_distance": min_tp1_distance,
            "note": "Active bullish advisory plan. BUY TP uses dedicated ATR targets.",
        }

    if action == "SELL PLAN" and bool(settings.get("short_plans_enabled", False)):
        stop = max(float(market.swing_high), price + stop_mult * atr)
        risk = max(stop - price, 0.01)

        sell_tp1_mult = float(settings.get("sell_tp1_atr_multiplier", 1.2))
        sell_tp2_mult = float(settings.get("sell_tp2_atr_multiplier", 2.0))
        support_buffer = float(settings.get("sell_support_buffer_atr", 0.15))

        entry_low = price - 0.10 * atr
        entry_high = price + 0.20 * atr
        tp1_raw = price - sell_tp1_mult * atr
        tp2_raw = price - sell_tp2_mult * atr
        tp1 = _sell_target_before_support(price, tp1_raw, float(market.swing_low), atr, support_buffer)
        tp2 = _sell_target_before_support(price, tp2_raw, float(market.support), atr, support_buffer)
        tp1 = _floor_below_price(price, tp1, 0.50 * atr)
        tp2 = min(_floor_below_price(price, tp2, 1.00 * atr), tp1 - 0.50 * atr)
        tp1_distance_atr = _tp1_distance_atr("sell", entry_low, entry_high, tp1, atr)
        tp1_too_close = _tp1_too_close(tp1_distance_atr, settings)
        quantity = advisory_position_size(account_size, risk_pct, (entry_low + entry_high) / 2.0, stop)

        return {
            "side": "sell_advisory",
            "quantity": quantity,
            "entry_zone_low": entry_low,
            "entry_zone_high": entry_high,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "risk": risk,
            "tp1_distance_atr": tp1_distance_atr,
            "tp1_too_close": tp1_too_close,
            "min_tp1_atr_distance": min_tp1_distance,
            "note": "Active bearish advisory plan. SELL TP uses reachable ATR/support-aware targets.",
        }

    if action == "EXIT LONG / AVOID BUY":
        return {
            "side": "exit_or_avoid",
            "quantity": 0.0,
            "entry_zone_low": None,
            "entry_zone_high": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
            "risk": 0.0,
            "tp1_distance_atr": 0.0,
            "tp1_too_close": False,
            "min_tp1_atr_distance": min_tp1_distance,
            "note": "Bearish conditions detected. This is not a short entry unless short plans are enabled.",
        }

    return {
        "side": "none",
        "quantity": 0.0,
        "entry_zone_low": None,
        "entry_zone_high": None,
        "stop": None,
        "tp1": None,
        "tp2": None,
        "risk": 0.0,
        "tp1_distance_atr": 0.0,
        "tp1_too_close": False,
        "min_tp1_atr_distance": min_tp1_distance,
        "note": "No active trade plan. Wait for a clean actionable signal.",
    }


def room_ratio(action: str, plan: dict, market) -> float:
    risk = float(plan.get("risk", 0) or 0)
    if risk <= 0:
        return 0.0
    if action == "BUY PLAN" and plan.get("entry_zone_high") is not None:
        room = float(market.resistance) - float(plan["entry_zone_high"])
    elif action == "SELL PLAN" and plan.get("entry_zone_low") is not None:
        room = float(plan["entry_zone_low"]) - float(market.support)
    else:
        return 0.0
    return max(room / risk, 0.0)


def advisory_position_size(account_equity: float, risk_pct: float, entry: float, stop: float) -> float:
    """Return advisory units only. This function never places orders."""
    if entry is None or stop is None:
        return 0.0
    risk_per_unit = abs(float(entry) - float(stop))
    if risk_per_unit <= 0:
        return 0.0
    raw_quantity = (float(account_equity) * float(risk_pct)) / risk_per_unit
    return math.floor(raw_quantity * 100) / 100
