from __future__ import annotations


def macro_context(manual_bias: str, news_block: bool) -> dict:
    bias = manual_bias if manual_bias in {"supportive", "restrictive", "mixed"} else "mixed"
    score = 0
    if bias == "supportive":
        score = 20
    elif bias == "restrictive":
        score = -20
    if news_block:
        return {"bias": bias, "score": score, "blocked": True, "note": "manual event block active"}
    return {"bias": bias, "score": score, "blocked": False, "note": "manual macro context"}
