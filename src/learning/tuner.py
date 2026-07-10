from __future__ import annotations

from copy import deepcopy
from itertools import chain

import pandas as pd
import yaml

from src.horizon_forecast import build_multi_horizon_forecast

from .github_store import GitHubLearningStore
from .schema import settings_hash, settings_snapshot


TUNABLE_KEYS = [
    "horizon_ridge_alpha",
    "horizon_model_lookback_bars",
    "horizon_min_probability_gap",
    "filter_min_probability_gap",
    "walk_forward_min_accuracy",
    "walk_forward_min_side_samples",
    "horizon_cost_points",
]


def _candidate_values(settings: dict, available_bars: int) -> dict:
    max_lookback = max(min(available_bars - 100, 5000), 500)
    return {
        "horizon_ridge_alpha": sorted({0.5, 1.0, 2.0, 5.0, float(settings.get("horizon_ridge_alpha", 2.0))}),
        "horizon_model_lookback_bars": sorted({500, 1000, 1500, 3000, max_lookback, int(settings.get("horizon_model_lookback_bars", 3000))}),
        "horizon_min_probability_gap": sorted({0.04, 0.06, 0.08, 0.10, float(settings.get("horizon_min_probability_gap", 0.06))}),
        "filter_min_probability_gap": sorted({0.06, 0.08, 0.10, 0.12, float(settings.get("filter_min_probability_gap", 0.08))}),
        "walk_forward_min_accuracy": sorted({55.0, 58.0, 60.0, float(settings.get("walk_forward_min_accuracy", 55.0))}),
        "walk_forward_min_side_samples": sorted({8, 12, 20, int(settings.get("walk_forward_min_side_samples", 8))}),
        "horizon_cost_points": sorted({0.20, 0.30, 0.50, float(settings.get("horizon_cost_points", 0.30))}),
    }


def _candidate_grid(settings: dict, available_bars: int) -> list[dict]:
    """Professional bounded search: current settings plus one-parameter-at-a-time challengers.

    Full Cartesian search is intentionally avoided inside Streamlit because it is slow and
    increases overfit risk. Each challenger must still pass the model's walk-forward validation.
    """
    candidates = [deepcopy(settings)]
    values = _candidate_values(settings, available_bars)
    for key, choices in values.items():
        for value in choices:
            candidate = deepcopy(settings)
            candidate[key] = value
            candidates.append(candidate)
    # Deduplicate by tunable snapshot.
    seen: set[tuple] = set()
    unique: list[dict] = []
    for candidate in candidates:
        signature = tuple((key, candidate.get(key)) for key in TUNABLE_KEYS)
        if signature not in seen:
            seen.add(signature)
            unique.append(candidate)
    return unique


def _score_forecasts(forecasts: list[dict]) -> tuple[float, dict]:
    by_horizon = {str(item.get("horizon")): item for item in forecasts}
    signal = by_horizon.get("5m", {})
    active_accuracy = signal.get("active_accuracy")
    if active_accuracy is None:
        return -1.0, {"reason": "5m active accuracy unavailable"}
    validation_tests = int(signal.get("validation_tests", 0) or 0)
    active_samples = int(signal.get("active_samples", 0) or 0)
    mae = float(signal.get("mae_points", 0.0) or 0.0)
    bull = signal.get("bull_confidence") or 0.0
    bear = signal.get("bear_confidence") or 0.0

    filter_bonus = 0.0
    for label in ["1h", "4h", "Daily"]:
        item = by_horizon.get(label, {})
        if item.get("validation_status") == "Validated":
            filter_bonus += 1.0

    score = (
        float(active_accuracy)
        + 0.10 * float(bull)
        + 0.10 * float(bear)
        + filter_bonus
        - 0.25 * mae
    )
    if validation_tests < 30 or active_samples < 8:
        score -= 20.0
    return round(score, 4), {
        "5m_active_accuracy": active_accuracy,
        "5m_active_samples": active_samples,
        "5m_validation_tests": validation_tests,
        "5m_bull_confidence": bull,
        "5m_bear_confidence": bear,
        "5m_mae_points": mae,
        "filter_bonus": filter_bonus,
    }


def generate_challenger_from_bars(bars: pd.DataFrame, settings: dict, store: GitHubLearningStore | None = None) -> dict:
    store = store or GitHubLearningStore.from_runtime()
    if bars is None or bars.empty or len(bars) < 500:
        return {"ok": False, "reason": "Need at least 500 clean bars before tuning."}

    candidates = _candidate_grid(settings, len(bars))
    scored: list[dict] = []
    for candidate in candidates:
        try:
            forecasts = build_multi_horizon_forecast(bars, candidate)
            score, detail = _score_forecasts(forecasts)
        except Exception as exc:
            score, detail = -1.0, {"reason": str(exc)}
        scored.append({"score": score, "detail": detail, "settings": {key: candidate.get(key) for key in TUNABLE_KEYS}})

    scored = sorted(scored, key=lambda item: item["score"], reverse=True)
    best = scored[0] if scored else None
    if not best or best["score"] <= 0:
        return {"ok": False, "reason": "No challenger passed validation scoring.", "tested": len(scored), "top": scored[:5]}

    challenger_settings = deepcopy(settings)
    challenger_settings.update(best["settings"])
    payload = {
        "schema_version": 1,
        "settings_hash": settings_hash(challenger_settings),
        "source": "walk_forward_challenger_search",
        "score": best["score"],
        "score_detail": best["detail"],
        "tuned_settings": best["settings"],
        "current_settings_hash": settings_hash(settings),
        "current_settings_snapshot": settings_snapshot(settings),
        "tested_candidates": len(scored),
        "top_candidates": scored[:10],
        "promotion_rule": "Manual review required. Promote only if challenger remains better after more out-of-sample outcomes.",
    }

    yaml_text = yaml.safe_dump(payload, sort_keys=False)
    write_result = {"ok": False, "reason": "GitHub learning store is not configured"}
    if store.enabled:
        write_result = store.write_text("config/challenger_settings.yaml", yaml_text, "learning: generate challenger settings")
        report = _challenger_report(payload)
        store.write_text("learning/reports/latest_learning_report.md", report, "learning: update challenger report")
    return {"ok": True, "tested": len(scored), "challenger": payload, "write": write_result}


def _challenger_report(payload: dict) -> str:
    lines = [
        "# Challenger Settings Report",
        "",
        f"Score: {payload.get('score')}",
        f"Settings hash: {payload.get('settings_hash')}",
        "",
        "## Tuned Settings",
        "",
        "```yaml",
        yaml.safe_dump(payload.get("tuned_settings", {}), sort_keys=False).strip(),
        "```",
        "",
        "## Score Detail",
        "",
        "```yaml",
        yaml.safe_dump(payload.get("score_detail", {}), sort_keys=False).strip(),
        "```",
        "",
        "Manual review required before promotion.",
    ]
    return "\n".join(lines) + "\n"
