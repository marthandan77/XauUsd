from __future__ import annotations

from src.gate_diagnostics import build_gate_breakdown, signal_stage, summarize_gate_blockers

from .backfill import backfill_learning_history as _backfill_learning_history
from .github_store import GitHubLearningStore
from .metrics import build_metrics_payload
from .outcome_evaluator import evaluate_due_predictions
from .prediction_logger import log_prediction


def run_learning_cycle(result: dict) -> dict:
    """Log current scan, evaluate due predictions, attach diagnostics, and return metrics.

    This function must never block trading decisions. It returns status instead of raising
    whenever the GitHub learning store is not configured.
    """
    if result and result.get("ok"):
        result["signal_stage"] = signal_stage(result)
        result["gate_breakdown"] = build_gate_breakdown(result)
        result["gate_summary"] = summarize_gate_blockers(result)

    store = GitHubLearningStore.from_runtime()
    if not store.enabled:
        return {
            "ok": False,
            "reason": "GitHub learning store is not configured",
            "store": store.status(),
            "signal_stage": result.get("signal_stage") if result else None,
            "gate_summary": result.get("gate_summary") if result else None,
            "gate_breakdown": result.get("gate_breakdown") if result else [],
        }

    log_status = log_prediction(result, store)
    eval_status = evaluate_due_predictions(result, store)
    metrics_status = build_metrics_payload(store, max_files=500)
    return {
        "ok": True,
        "store": store.status(),
        "signal_stage": result.get("signal_stage") if result else None,
        "gate_summary": result.get("gate_summary") if result else None,
        "gate_breakdown": result.get("gate_breakdown") if result else [],
        "log": log_status,
        "evaluation": eval_status,
        "metrics": metrics_status,
    }


def refresh_learning_metrics() -> dict:
    store = GitHubLearningStore.from_runtime()
    return build_metrics_payload(store, max_files=500)


def generate_challenger(result: dict) -> dict:
    """Disabled by design to reduce overfitting."""
    return {
        "ok": False,
        "reason": "Challenger/recommended settings generation is disabled to reduce overfitting.",
        "new_direction": "Use gate breakdown, fixed champion settings, and out-of-sample validation before any setting change.",
    }


def backfill_learning_history(result: dict) -> dict:
    if not result or not result.get("ok"):
        return {"ok": False, "reason": "Run SCAN NOW before backfill."}
    return _backfill_learning_history(result)
