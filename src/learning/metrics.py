from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .github_store import GitHubLearningStore


def load_outcomes(store: GitHubLearningStore | None = None, max_files: int = 500) -> list[dict]:
    store = store or GitHubLearningStore.from_runtime()
    if not store.enabled:
        return []
    outcomes: list[dict] = []
    for path in store.list_json_files("learning/outcomes", max_files=max_files):
        item = store.read_json(path)
        if item:
            outcomes.append(item)
    return outcomes


def _pct(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100.0, 1)


def compute_learning_metrics(outcomes: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in outcomes:
        groups[str(item.get("horizon", "unknown"))].append(item)

    rows: list[dict] = []
    order = {"5m": 0, "1h": 1, "4h": 2, "Daily": 3}
    for horizon, items in sorted(groups.items(), key=lambda pair: order.get(pair[0], 99)):
        active = [x for x in items if x.get("predicted_bias") in {"bullish", "bearish"}]
        bull = [x for x in items if x.get("predicted_bias") == "bullish"]
        bear = [x for x in items if x.get("predicted_bias") == "bearish"]
        mixed = [x for x in items if x.get("predicted_bias") == "mixed"]
        active_correct = sum(1 for x in active if x.get("correct") is True)
        bull_correct = sum(1 for x in bull if x.get("correct") is True)
        bear_correct = sum(1 for x in bear if x.get("correct") is True)
        mixed_correct = sum(1 for x in mixed if x.get("correct") is True)
        active_points = [float(x.get("realized_points", 0.0) or 0.0) for x in active]
        false_positives = len(active) - active_correct
        rows.append(
            {
                "Horizon": horizon,
                "Completed": len(items),
                "Active Samples": len(active),
                "Bull Samples": len(bull),
                "Bull Accuracy": _pct(bull_correct, len(bull)),
                "Bear Samples": len(bear),
                "Bear Accuracy": _pct(bear_correct, len(bear)),
                "Mixed Samples": len(mixed),
                "Mixed Accuracy": _pct(mixed_correct, len(mixed)),
                "Active Accuracy": _pct(active_correct, len(active)),
                "False Positive Rate": _pct(false_positives, len(active)),
                "Avg Active Points": round(mean(active_points), 2) if active_points else None,
            }
        )
    return rows


def learning_report_markdown(metrics: list[dict]) -> str:
    lines = ["# XAU/USD Learning Report", "", "Generated from completed prediction outcomes.", ""]
    if not metrics:
        lines.append("No completed outcomes yet.")
        return "\n".join(lines) + "\n"
    lines.append("| Horizon | Completed | Active Accuracy | Bull Accuracy | Bear Accuracy | Avg Active Points | False Positive Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in metrics:
        lines.append(
            f"| {row['Horizon']} | {row['Completed']} | {row['Active Accuracy']} | {row['Bull Accuracy']} | {row['Bear Accuracy']} | {row['Avg Active Points']} | {row['False Positive Rate']} |"
        )
    lines.append("")
    lines.append("Promotion rule: do not apply challenger settings unless sample count is sufficient and active accuracy, false positives, and realized points improve together.")
    return "\n".join(lines) + "\n"


def build_metrics_payload(store: GitHubLearningStore | None = None, max_files: int = 500) -> dict:
    store = store or GitHubLearningStore.from_runtime()
    outcomes = load_outcomes(store, max_files=max_files)
    metrics = compute_learning_metrics(outcomes)
    return {"ok": bool(store.enabled), "store": store.status(), "outcome_count": len(outcomes), "metrics": metrics, "report": learning_report_markdown(metrics)}
