from __future__ import annotations


def build_trade_plan(action: str, market, settings: dict) -> dict:
    price = float(market.price)
    atr = float(market.atr)
    mult = float(settings.get("atr_multiplier", 1.3))
    tp1_r = float(settings.get("tp1_r", 1.5))
    tp2_r = float(settings.get("tp2_r", 2.0))
    if action == "BUY PLAN":
        stop = min(float(market.swing_low), price - mult * atr)
        risk = max(price - stop, 0.01)
        return {"side": "buy", "entry_zone_low": price - 0.20 * atr, "entry_zone_high": price + 0.10 * atr, "stop": stop, "tp1": price + tp1_r * risk, "tp2": price + tp2_r * risk, "risk": risk}
    if action == "SELL PLAN":
        stop = max(float(market.swing_high), price + mult * atr)
        risk = max(stop - price, 0.01)
        return {"side": "sell", "entry_zone_low": price - 0.10 * atr, "entry_zone_high": price + 0.20 * atr, "stop": stop, "tp1": price - tp1_r * risk, "tp2": price - tp2_r * risk, "risk": risk}
    return {"side": "none", "entry_zone_low": price, "entry_zone_high": price, "stop": price, "tp1": price, "tp2": price, "risk": 0.0}


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
