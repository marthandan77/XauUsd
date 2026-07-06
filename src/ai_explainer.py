from __future__ import annotations


def explain(regime: str, scores: dict, veto: dict, market, macro: dict) -> str:
    text = [
        f"Regime: {regime}.",
        f"Bias: {scores.get('bias')} with confidence {scores.get('confidence')}.",
        f"Location: {market.range_position_pct:.1f}% of current range.",
        f"Macro context: {macro.get('bias')}.",
    ]

    kc_state = scores.get("kc_state")
    if kc_state and kc_state != "not_available":
        text.append(f"KC Squeeze state: {kc_state}.")
    kc_reason = scores.get("kc_reason")
    if kc_reason:
        text.append(kc_reason)
    notes = scores.get("notes") or []
    if notes:
        text.append("Signal notes: " + "; ".join(dict.fromkeys(notes)) + ".")

    reasons = veto.get("reasons") or []
    if reasons:
        text.append("Rejected because: " + "; ".join(reasons) + ".")
    else:
        text.append("No hard rejection was triggered.")
    return " ".join(text)
