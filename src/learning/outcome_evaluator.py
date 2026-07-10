from __future__ import annotations

import math
from datetime import date

import pandas as pd

from .github_store import GitHubLearningStore
from .schema import HORIZON_MINUTES, actual_label, horizon_due_time, json_safe, realized_points_for_bias


def _bars_index_in_timezone(bars: pd.DataFrame, timezone: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(bars.index)
    if index.tz is None:
        return index.tz_localize(timezone)
    return index.tz_convert(timezone)


def _price_at_or_after(bars: pd.DataFrame, timestamp, timezone: str):
    if bars is None or bars.empty:
        return None, None
    index = _bars_index_in_timezone(bars, timezone)
    target = pd.Timestamp(timestamp)
    if target.tzinfo is None:
        target = target.tz_localize(timezone)
    else:
        target = target.tz_convert(timezone)
    pos = index.searchsorted(target, side="left")
    if pos >= len(index):
        return None, None
    price = float(bars.iloc[int(pos)]["close"])
    return price, index[int(pos)]


def _outcome_path(scan_time: str, scan_id: str, horizon: str) -> str:
    day = str(scan_time)[:10] or date.today().isoformat()
    safe_horizon = horizon.replace("/", "_").replace(" ", "_")
    return f"learning/outcomes/{day}/{scan_id}_{safe_horizon}.json"


def _build_outcome(event: dict, horizon_event: dict, bars: pd.DataFrame, settings: dict):
    horizon = str(horizon_event.get("horizon"))
    if horizon not in HORIZON_MINUTES:
        return None
    scan_time = pd.Timestamp(event.get("scan_time"))
    timezone = str(settings.get("timezone", "Asia/Singapore"))
    due_time = horizon_due_time(scan_time, horizon)
    now = pd.Timestamp.now(tz=timezone)
    if now < due_time:
        return None

    price_at_scan = float(event.get("price_at_scan", 0.0) or 0.0)
    price_after, outcome_time = _price_at_or_after(bars, due_time, timezone)
    if price_after is None or outcome_time is None or price_at_scan <= 0:
        return None

    cost_points = float(event.get("settings_used", {}).get("horizon_cost_points", settings.get("horizon_cost_points", 0.30)))
    predicted_bias = str(horizon_event.get("bias", "mixed"))
    actual = actual_label(price_at_scan, price_after, cost_points)
    return {
        "schema_version": 1,
        "scan_id": event.get("scan_id"),
        "scan_time": event.get("scan_time"),
        "outcome_time": json_safe(outcome_time),
        "symbol": event.get("symbol"),
        "timeframe": event.get("timeframe"),
        "horizon": horizon,
        "price_at_scan": price_at_scan,
        "price_after": round(float(price_after), 4),
        "predicted_bias": predicted_bias,
        "actual_label": actual,
        "correct": bool(predicted_bias == actual),
        "realized_points": realized_points_for_bias(predicted_bias, price_at_scan, price_after),
        "bull_confidence": horizon_event.get("bull_confidence"),
        "bear_confidence": horizon_event.get("bear_confidence"),
        "validation_status": horizon_event.get("validation_status"),
        "settings_hash": event.get("settings_hash"),
    }


def evaluate_due_predictions(result: dict, store: GitHubLearningStore | None = None, max_prediction_files: int = 120) -> dict:
    store = store or GitHubLearningStore.from_runtime()
    if not store.enabled:
        return {"ok": False, "stage": "evaluate_due_predictions", "reason": "GitHub learning store is not configured", "store": store.status()}
    if not result.get("ok"):
        return {"ok": False, "stage": "evaluate_due_predictions", "reason": "scan result is not ok"}

    bars = result.get("bars")
    settings = result.get("settings", {})
    prediction_paths = store.list_json_files("learning/predictions", max_files=max_prediction_files)
    checked = 0
    written = 0
    skipped_existing = 0
    skipped_not_due = 0

    for path in prediction_paths:
        event = store.read_json(path)
        if not event:
            continue
        for horizon_event in event.get("horizons", []):
            horizon = str(horizon_event.get("horizon"))
            outcome_path = _outcome_path(str(event.get("scan_time", "")), str(event.get("scan_id")), horizon)
            if store.read_json(outcome_path):
                skipped_existing += 1
                continue
            outcome = _build_outcome(event, horizon_event, bars, settings)
            if outcome is None:
                skipped_not_due += 1
                continue
            checked += 1
            write = store.write_json(outcome_path, outcome, f"learning: outcome {outcome['scan_id']} {horizon}")
            if write.get("ok"):
                written += 1

    return {
        "ok": True,
        "stage": "evaluate_due_predictions",
        "prediction_files": len(prediction_paths),
        "checked": checked,
        "written": written,
        "skipped_existing": skipped_existing,
        "skipped_not_due": skipped_not_due,
    }
