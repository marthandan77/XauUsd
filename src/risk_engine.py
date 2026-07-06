from __future__ import annotations


def calculate_position_size(account_equity: float, risk_pct: float, entry: float, stop_loss: float) -> float:
    """Calculate advisory units only. This does not place broker orders."""
    risk_per_unit = abs(float(entry) - float(stop_loss))
    if risk_per_unit <= 0:
        return 0.0
    return (float(account_equity) * float(risk_pct)) / risk_per_unit


def calculate_risk_reward(entry: float, stop_loss: float, take_profit: float) -> float:
    risk = abs(float(entry) - float(stop_loss))
    reward = abs(float(take_profit) - float(entry))
    if risk <= 0:
        return 0.0
    return reward / risk
