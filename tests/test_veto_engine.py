from types import SimpleNamespace

from src.veto_engine import apply_veto


def market(**kwargs):
    base = dict(
        atr=10.0,
        middle_range=False,
        near_resistance=False,
        near_support=False,
        resistance=2400.0,
        support=2300.0,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_event_block_rejects_plan():
    plan = {"risk": 10, "entry_zone_high": 2350, "entry_zone_low": 2349}
    out = apply_veto("BUY PLAN", plan, market(), "bull_trend", {"blocked": True}, {"min_reward_risk": 1.5})
    assert out["final_action"] == "HOLD"
    assert "event block active" in out["reasons"]


def test_middle_range_rejects_plan():
    plan = {"risk": 10, "entry_zone_high": 2350, "entry_zone_low": 2349}
    out = apply_veto("BUY PLAN", plan, market(middle_range=True), "bull_trend", {"blocked": False}, {"min_reward_risk": 1.5})
    assert out["final_action"] == "HOLD"


def test_wait_is_not_forced_to_plan():
    plan = {"risk": 0, "entry_zone_high": 2350, "entry_zone_low": 2350}
    out = apply_veto("WAIT", plan, market(), "range", {"blocked": False}, {"min_reward_risk": 1.5})
    assert out["final_action"] == "WAIT"
