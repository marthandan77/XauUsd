from __future__ import annotations


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
        }

    if action == "SELL PLAN" and bool(settings.get("short_plans_enabled", False)):
        stop = max(float(market.swing_high), price + stop_mult * atr)
        risk = max(stop - price, 0.01)
        tp1 = price - min(tp_mult, tp1_r * stop_mult) * atr
        tp2 = price - max(tp_mult, tp2_r * stop_mult) * atr
        return {
            "side": "sell_advisory",
            "entry_zone_low": price - 0.10 * atr,
            "entry_zone_high": price + 0.20 * atr,
            "stop": stop,
            "tp1": tp1,
            "tp2": tp2,
            "risk": risk,
        }

    return {
        "side": "none",
        "entry_zone_low": price,
        "entry_zone_high": price,
        "stop": price,
        "tp1": price,
        "tp2": price,
        "risk": 0.0,
    }


def room_ratio(action: str, plan: dict, market) -> float:
    risk = float(plan.get("risk", 0) or 0)
    if risk <= 0:
        return 0.0
    if action == "BUY PLAN":
        room = float(market.resistance) - float(plan["entry_zone_high"])
    elif action == "SELL PLAN":
        room = float(plan["entry_zone_low"]) - float(market.support)
    else:
        return 0.0
    return max(room / risk, 0.0)


def advisory_position_size(account_equity: float, risk_pct: float, entry: float, stop: float) -> float:
    """Return advisory units only. This function never places orders."""
    risk_per_unit = abs(float(entry) - float(stop))
    if risk_per_unit <= 0:
        return 0.0
    return (float(account_equity) * float(risk_pct)) / risk_per_unit
