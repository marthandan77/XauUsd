from __future__ import annotations


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


def build_trade_plan(action: str, market, settings: dict) -> dict:
    price = float(market.price)
    atr = float(market.atr)
    stop_mult = float(settings.get("atr_stop_multiplier", settings.get("atr_multiplier", 1.3)))
    tp_mult = float(settings.get("atr_tp_multiplier", 2.0))
    tp1_r = float(settings.get("tp1_r", 1.5))
    tp2_r = float(settings.get("tp2_r", 2.0))

    if action == "BUY PLAN":
        stop = min(float(market.swing_low), price - stop_mult * atr)
        risk = max(price - stop, 0.01)
        tp1 = price + min(tp_mult, tp1_r * stop_mult) * atr
        tp2 = price + max(tp_mult, tp2_r * stop_mult) * atr
        return {
            "side": "buy_advisory",
            "entry_zone_low": price - 0.20 * atr,
            "entry_zone_high": price + 0.10 * atr,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "risk": risk,
            "note": "Active bullish advisory plan.",
        }

    if action == "SELL PLAN" and bool(settings.get("short_plans_enabled", False)):
        stop = max(float(market.swing_high), price + stop_mult * atr)
        risk = max(stop - price, 0.01)

        # Short targets are intentionally tighter than the old generic ATR target.
        # The old formula used 4.5 ATR for TP1 and 6.0 ATR for TP2 with default settings,
        # which was too far for XAU/USD scalp-style bearish plans.
        sell_tp1_mult = float(settings.get("sell_tp1_atr_multiplier", 1.25))
        sell_tp2_mult = float(settings.get("sell_tp2_atr_multiplier", 2.25))
        support_buffer = float(settings.get("sell_support_buffer_atr", 0.15))

        tp1_raw = price - sell_tp1_mult * atr
        tp2_raw = price - sell_tp2_mult * atr
        tp1 = _sell_target_before_support(price, tp1_raw, float(market.swing_low), atr, support_buffer)
        tp2 = _sell_target_before_support(price, tp2_raw, float(market.support), atr, support_buffer)
        tp1 = _floor_below_price(price, tp1, 0.50 * atr)
        tp2 = min(_floor_below_price(price, tp2, 1.00 * atr), tp1 - 0.50 * atr)

        return {
            "side": "sell_advisory",
            "entry_zone_low": price - 0.10 * atr,
            "entry_zone_high": price + 0.20 * atr,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "risk": risk,
            "note": "Active bearish advisory plan. SELL TP uses reachable ATR/support-aware targets.",
        }

    if action == "EXIT LONG / AVOID BUY":
        return {
            "side": "exit_or_avoid",
            "entry_zone_low": None,
            "entry_zone_high": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
            "risk": 0.0,
            "note": "Bearish conditions detected. This is not a short entry unless short plans are enabled.",
        }

    return {
        "side": "none",
        "entry_zone_low": None,
        "entry_zone_high": None,
        "stop": None,
        "tp1": None,
        "tp2": None,
        "risk": 0.0,
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
    return (float(account_equity) * float(risk_pct)) / risk_per_unit
