from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.horizon_forecast import (
    HORIZON_CONFIG,
    _base_minutes,
    _bias_from_probs,
    _fallback_sigma,
    _feature_frame,
    _norm_cdf,
    _ridge_fit_predict,
)

from .github_store import GitHubLearningStore
from .schema import actual_label, json_safe, realized_points_for_bias, settings_hash, settings_snapshot


def _timestamp(value) -> str:
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _candidate_indices(length: int, steps: int, min_train: int, max_points: int) -> list[int]:
    last_idx = length - steps - 1
    if last_idx <= min_train:
        return []
    start = max(min_train, last_idx - max_points + 1)
    return list(range(start, last_idx + 1))


def _build_backfill_outcomes_for_horizon(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    settings: dict,
    label: str,
    config: dict,
    max_points: int,
) -> list[dict[str, Any]]:
    horizon_minutes = int(config["minutes"])
    base_minutes = _base_minutes(settings)
    if base_minutes <= 0 or base_minutes > horizon_minutes or horizon_minutes % base_minutes != 0:
        return []

    steps = max(int(horizon_minutes / base_minutes), 1)
    close = bars["close"].astype(float)
    target = np.log(close.shift(-steps) / close).replace([np.inf, -np.inf], np.nan)
    min_train = int(settings.get("walk_forward_min_train", 300))
    lookback = int(settings.get("horizon_model_lookback_bars", 3000))
    alpha = float(settings.get("horizon_ridge_alpha", 2.0))
    cost_points = float(settings.get("horizon_cost_points", 0.30))

    outcomes: list[dict[str, Any]] = []
    for idx in _candidate_indices(len(bars), steps, min_train, max_points):
        price_at_scan = float(close.iloc[idx])
        price_after = float(close.iloc[idx + steps])
        if price_at_scan <= 0 or price_after <= 0:
            continue
        train_start = max(0, idx - lookback)
        x_train = features.iloc[train_start:idx].to_numpy(dtype=float)
        y_train = target.iloc[train_start:idx].to_numpy(dtype=float)
        x_now = features.iloc[idx].to_numpy(dtype=float)
        y_now = target.iloc[idx]
        if not math.isfinite(float(y_now)):
            continue

        mu, resid_std, training_rows = _ridge_fit_predict(x_train, y_train, x_now, alpha=alpha)
        if training_rows < min_train:
            continue
        sigma = resid_std if math.isfinite(float(resid_std)) and float(resid_std) > 0 else _fallback_sigma(pd.Series(y_train).dropna(), steps)
        sigma = max(float(sigma), 1e-6)
        cost_return = cost_points / max(price_at_scan, 0.01)
        p_up = 1.0 - _norm_cdf((cost_return - mu) / sigma)
        p_down = _norm_cdf((-cost_return - mu) / sigma)
        predicted_bias = _bias_from_probs(p_up, p_down, settings, str(config.get("mode", "filter")))
        actual = actual_label(price_at_scan, price_after, cost_points)

        scan_time = bars.index[idx]
        outcome_time = bars.index[idx + steps]
        outcomes.append(
            {
                "schema_version": 1,
                "source": "historical_backfill",
                "scan_id": f"backfill_{pd.Timestamp(scan_time).strftime('%Y%m%d_%H%M%S')}_{label}",
                "scan_time": _timestamp(scan_time),
                "outcome_time": _timestamp(outcome_time),
                "symbol": str(settings.get("twelve_data_symbol", "XAU/USD")),
                "timeframe": str(settings.get("price_interval", "5m")),
                "horizon": label,
                "price_at_scan": round(price_at_scan, 4),
                "price_after": round(price_after, 4),
                "predicted_bias": predicted_bias,
                "actual_label": actual,
                "correct": bool(predicted_bias == actual),
                "realized_points": realized_points_for_bias(predicted_bias, price_at_scan, price_after),
                "model_up_probability": round(p_up * 100.0, 1),
                "model_down_probability": round(p_down * 100.0, 1),
                "expected_return_pct": round((math.exp(mu) - 1.0) * 100.0, 4),
                "training_rows": training_rows,
                "settings_hash": settings_hash(settings),
            }
        )
    return outcomes


def build_backfill_payload(bars: pd.DataFrame, settings: dict, max_points: int | None = None) -> dict:
    if bars is None or bars.empty:
        return {"ok": False, "reason": "No bars available. Run SCAN NOW first."}
    clean = bars.dropna(subset=["open", "high", "low", "close"]).copy()
    if len(clean) < int(settings.get("walk_forward_min_train", 300)) + 20:
        return {"ok": False, "reason": "Not enough clean bars for historical backfill."}

    max_points = int(max_points or settings.get("learning_backfill_points", 60))
    max_points = max(5, min(max_points, 250))
    features = _feature_frame(clean, settings)
    outcomes: list[dict] = []
    for label, config in HORIZON_CONFIG.items():
        outcomes.extend(_build_backfill_outcomes_for_horizon(clean, features, settings, label, config, max_points))

    payload = {
        "schema_version": 1,
        "source": "historical_backfill_batch",
        "generated_at": pd.Timestamp.now(tz=str(settings.get("timezone", "Asia/Singapore"))).isoformat(),
        "settings_hash": settings_hash(settings),
        "settings_snapshot": settings_snapshot(settings),
        "bars_used": len(clean),
        "max_points_per_horizon": max_points,
        "outcome_count": len(outcomes),
        "outcomes": outcomes,
    }
    return {"ok": True, "payload": payload}


def backfill_learning_history(result: dict, store: GitHubLearningStore | None = None, max_points: int | None = None) -> dict:
    store = store or GitHubLearningStore.from_runtime()
    if not store.enabled:
        return {"ok": False, "stage": "backfill_learning_history", "reason": "GitHub learning store is not configured", "store": store.status()}
    if not result or not result.get("ok"):
        return {"ok": False, "stage": "backfill_learning_history", "reason": "Run SCAN NOW before backfill."}

    built = build_backfill_payload(result.get("bars"), result.get("settings", {}), max_points=max_points)
    if not built.get("ok"):
        built["stage"] = "backfill_learning_history"
        return built

    payload = built["payload"]
    day = str(payload.get("generated_at", ""))[:10] or date.today().isoformat()
    stamp = pd.Timestamp(payload["generated_at"]).strftime("%Y%m%d_%H%M%S")
    path = f"learning/backfills/{day}/backfill_{stamp}.json"
    write = store.write_json(path, json_safe(payload), f"learning: backfill {payload['outcome_count']} outcomes")
    return {
        "ok": bool(write.get("ok")),
        "stage": "backfill_learning_history",
        "path": path,
        "outcome_count": payload.get("outcome_count", 0),
        "bars_used": payload.get("bars_used", 0),
        "write": write,
    }
