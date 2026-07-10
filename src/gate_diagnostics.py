from __future__ import annotations

from typing import Any


ACTIONABLE = {"BUY PLAN", "SELL PLAN"}


def _horizon_map(result: dict) -> dict[str, dict]:
    return {str(item.get("horizon")): item for item in result.get("multi_horizon", []) if isinstance(item, dict)}


def _status(pass_condition: bool, fail_condition: bool = False) -> str:
    if pass_condition:
        return "PASS"
    if fail_condition:
        return "BLOCK"
    return "WATCH"


def _row(gate: str, status: str, detail: str, blocking: bool) -> dict[str, Any]:
    return {"Gate": gate, "Status": status, "Blocking": bool(blocking), "Detail": detail}


def signal_stage(result: dict) -> str:
    if not result or not result.get("ok"):
        return "NO SCAN"
    veto = result.get("veto", {})
    scores = result.get("scores", {})
    final_action = str(veto.get("final_action", "HOLD"))
    if final_action in ACTIONABLE:
        return "TRADE PERMISSION"
    if bool(scores.get("force_wait", False)):
        return "WAIT - FORCED"
    max_score = max(int(scores.get("bull_score", 0) or 0), int(scores.get("bear_score", 0) or 0))
    settings = result.get("settings", {})
    setup_threshold = int(settings.get("forecast_threshold", 65))
    watch_threshold = min(int(settings.get("wait_threshold", 60)), 45)
    if max_score >= setup_threshold:
        return "SCALP SETUP"
    if max_score >= watch_threshold:
        return "SCALP WATCH"
    return "NO EDGE"


def build_gate_breakdown(result: dict) -> list[dict[str, Any]]:
    if not result or not result.get("ok"):
        return [_row("Scan", "BLOCK", "No valid scan result. Press SCAN NOW first.", True)]

    settings = result.get("settings", {})
    scores = result.get("scores", {})
    veto = result.get("veto", {})
    market = result.get("market")
    plan = result.get("plan", {})
    action = str(result.get("action", "HOLD"))
    final_action = str(veto.get("final_action", "HOLD"))
    horizons = _horizon_map(result)
    bull = int(scores.get("bull_score", 0) or 0)
    bear = int(scores.get("bear_score", 0) or 0)
    buy_threshold = int(settings.get("buy_threshold", 78))
    sell_threshold = int(settings.get("sell_threshold", 78))
    max_score = max(bull, bear)
    direction = "bullish" if bull > bear else "bearish" if bear > bull else "mixed"

    rows: list[dict[str, Any]] = []

    data_block = bool(settings.get("signals_disabled", False))
    rows.append(_row(
        "Data",
        "BLOCK" if data_block else "PASS",
        str(settings.get("signals_disabled_reason", "Live/real data available.")) if data_block else "Real feed or uploaded real CSV is active.",
        data_block,
    ))

    force_wait = bool(scores.get("force_wait", False))
    rows.append(_row(
        "Forced wait",
        "BLOCK" if force_wait else "PASS",
        str(scores.get("wait_reason", "No forced wait condition.")) if force_wait else "No macro/KC forced wait condition.",
        force_wait,
    ))

    score_pass = (bull >= buy_threshold and bull > bear) or (bear >= sell_threshold and bear > bull)
    score_detail = f"Bull {bull}/{buy_threshold}; Bear {bear}/{sell_threshold}; current pressure = {direction}."
    rows.append(_row("Score", "PASS" if score_pass else "WATCH" if max_score >= 45 else "BLOCK", score_detail, not score_pass))

    if market is not None:
        middle = bool(getattr(market, "middle_range", False))
        location = float(getattr(market, "range_position_pct", 0.0) or 0.0)
        rows.append(_row(
            "Location",
            "BLOCK" if middle else "PASS",
            f"Range position {location:.1f}%. Middle range blocks final permission, but can still be a watch condition.",
            middle and action in ACTIONABLE,
        ))
    else:
        rows.append(_row("Location", "BLOCK", "Market map unavailable.", True))

    rr = float(veto.get("room_ratio", 0.0) or 0.0)
    min_rr = float(settings.get("min_reward_risk", 1.2))
    rows.append(_row(
        "Room / reward-risk",
        "PASS" if rr >= min_rr else "BLOCK",
        f"Room ratio {rr:.2f}; required {min_rr:.2f}. Zero usually means price is too close to target/support/resistance or no active plan exists.",
        action in ACTIONABLE and rr < min_rr,
    ))

    h5 = horizons.get("5m")
    if not h5:
        rows.append(_row("5m entry model", "BLOCK", "5m horizon unavailable.", action in ACTIONABLE))
    else:
        h5_bias = h5.get("bias")
        h5_decision = h5.get("decision")
        bull_acc = h5.get("bull_confidence")
        bear_acc = h5.get("bear_confidence")
        rows.append(_row(
            "5m entry model",
            "PASS" if str(h5_decision).startswith("Validated") else "WATCH" if h5_bias in {"bullish", "bearish"} else "BLOCK",
            f"Current read {h5_bias}; decision {h5_decision}; bull accuracy {bull_acc}; bear accuracy {bear_acc}.",
            action in ACTIONABLE and not str(h5_decision).startswith("Validated"),
        ))

    filter_details = []
    filter_block = False
    for label in ["1h", "4h", "Daily"]:
        item = horizons.get(label)
        if not item:
            filter_details.append(f"{label}: unavailable")
            filter_block = filter_block or action in ACTIONABLE
            continue
        bias = item.get("bias")
        decision = item.get("decision")
        validation = item.get("validation_status")
        filter_details.append(f"{label}: {bias}, {validation}, {decision}")
        if action == "BUY PLAN" and bias == "bearish":
            filter_block = True
        if action == "SELL PLAN" and bias == "bullish":
            filter_block = True
    rows.append(_row(
        "Higher timeframe filters",
        "BLOCK" if filter_block else "PASS",
        "; ".join(filter_details) if filter_details else "No filters available.",
        filter_block,
    ))

    veto_reasons = veto.get("reasons", []) or []
    rows.append(_row(
        "Final veto",
        "BLOCK" if veto_reasons else "PASS" if final_action in ACTIONABLE else "WATCH",
        "; ".join(veto_reasons) if veto_reasons else f"Final action is {final_action}.",
        bool(veto_reasons),
    ))

    return rows


def summarize_gate_blockers(result: dict) -> str:
    rows = build_gate_breakdown(result)
    blockers = [row for row in rows if row.get("Blocking")]
    if not blockers:
        return "No blocking veto. If no trade fired, the raw signal score did not reach trade permission."
    return " | ".join(f"{row['Gate']}: {row['Detail']}" for row in blockers)
