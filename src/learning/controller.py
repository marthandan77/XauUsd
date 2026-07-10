from __future__ import annotations

from src.gate_diagnostics import build_gate_breakdown, signal_stage, summarize_gate_blockers

from .backfill import backfill_learning_history as _backfill_learning_history
from .github_store import GitHubLearningStore
from .metrics import build_metrics_payload
from .outcome_evaluator import evaluate_due_predictions
from .prediction_logger import log_prediction


def run_learning_cycle(result: dict) -> dict:
    """Log current scan, evaluate due historical predictions, and return current metrics.

    This function must never block trading decisions. It returns status instead of raising
    whenever the GitHub learning store is not configured.
    """
    if result and result.get("ok"):
        result["signal_stage"] = signal_stage(result)
        result["gate_breakdown"] = build_gate_breakdown(result)
        result["gate_summary"] = summarize_gate_blockers(result)

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
    """Disabled by design.

    We removed automatic challenger/recommended settings because it encourages overfitting
    to recent backfill/live samples. The new direction is diagnostics-first: freeze the
    champion settings, inspect gate failures, then change one hypothesis at a time.
    """
    return {
        "ok": False,
        "reason": "Challenger/recommended settings generation is disabled to reduce overfitting.",
        "new_direction": "Use gate breakdown, fixed champion settings, and out-of-sample validation before any setting change.",
    }


def backfill_learning_history(result: dict) -> dict:
    if not result or not result.get("ok"):
        return {"ok": False, "reason": "Run SCAN NOW before backfill."}
    return _backfill_learning_history(result)


def _install_sidebar_backfill_button() -> None:
    """Install a safe sidebar backfill button below the existing Apply button.

    The main Streamlit app already imports this controller. This installer avoids a large
    app.py rewrite while keeping the backfill UI next to the settings controls. It does
    not execute any Streamlit command during import; it only wraps the button method and
    renders the backfill button when the existing settings Apply button is drawn.
    """
    try:
        import streamlit as st
        from streamlit.delta_generator import DeltaGenerator
    except Exception:
        return

    current = DeltaGenerator.button
    if getattr(current, "_xau_backfill_patched", False):
        return
    original = getattr(current, "_xau_original_button", current)

    def patched_button(self, label, *args, **kwargs):
        if label == "Generate challenger settings":
            self.info("Challenger/recommended settings generation has been removed to reduce overfitting. Use gate breakdown and fixed champion settings instead.")
            return False

        clicked = original(self, label, *args, **kwargs)
        if label == "Apply" and kwargs.get("key") == "settings_apply_button":
            result = st.session_state.get("last_scan_result")
            ready = isinstance(result, dict) and bool(result.get("ok")) and result.get("bars") is not None
            backfill_clicked = st.sidebar.button(
                "Backfill Learning History",
                key="learning_backfill_sidebar_button",
                disabled=not ready,
                use_container_width=True,
                help="Run SCAN NOW first. Backfill uses the latest scanned bars to create historical walk-forward outcome samples.",
            )
            st.sidebar.caption("Backfill is enabled only after SCAN NOW.")
            if backfill_clicked:
                with st.sidebar:
                    with st.spinner("Backfilling learning history..."):
                        status = backfill_learning_history(result)
                st.session_state["learning_backfill"] = status
                st.session_state["learning_metrics"] = refresh_learning_metrics()
                if status.get("ok"):
                    st.sidebar.success(f"Backfilled {status.get('outcome_count', 0)} outcomes.")
                else:
                    st.sidebar.warning(status.get("reason", "Backfill failed."))
        return clicked

    patched_button._xau_backfill_patched = True
    patched_button._xau_original_button = original
    DeltaGenerator.button = patched_button


_install_sidebar_backfill_button()
