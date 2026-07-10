from __future__ import annotations

from .github_store import GitHubLearningStore
from .metrics import build_metrics_payload
from .outcome_evaluator import evaluate_due_predictions
from .prediction_logger import log_prediction
from .tuner import generate_challenger_from_bars


def run_learning_cycle(result: dict) -> dict:
    """Log current scan, evaluate due historical predictions, and return current metrics.

    This function must never block trading decisions. It returns status instead of raising
    whenever the GitHub learning store is not configured.
    """
    store = GitHubLearningStore.from_runtime()
    status = {"store": store.status()}
    if not store.enabled:
        status.update({"ok": False, "reason": "GitHub learning store is not configured"})
        return status

    log_status = log_prediction(result, store)
    eval_status = evaluate_due_predictions(result, store)
    metrics_status = build_metrics_payload(store, max_files=500)
    return {
        "ok": True,
        "store": store.status(),
        "log": log_status,
        "evaluation": eval_status,
        "metrics": metrics_status,
    }


def refresh_learning_metrics() -> dict:
    store = GitHubLearningStore.from_runtime()
    return build_metrics_payload(store, max_files=500)


def generate_challenger(result: dict) -> dict:
    store = GitHubLearningStore.from_runtime()
    if not result or not result.get("ok"):
        return {"ok": False, "reason": "Run SCAN NOW before generating a challenger."}
    return generate_challenger_from_bars(result.get("bars"), result.get("settings", {}), store)
