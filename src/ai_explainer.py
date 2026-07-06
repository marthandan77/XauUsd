from __future__ import annotations


def explain(regime: str, scores: dict, veto: dict, market, macro: dict) -> str:
    text = [
        f"Regime: {regime}.",
        f"Bias: {scores.get('bias')} with confidence {scores.get('confidence')}.",
        f"Location: {market.range_position_pct:.1f}% of current range.",
        f"Macro context: {macro.get('bias')}.",
    ]
    reasons = veto.get("reasons") or []
    if reasons:
        text.append("Rejected because: " + "; ".join(reasons) + ".")
    else:
        text.append("No hard rejection was triggered.")
    return " ".join(text)
