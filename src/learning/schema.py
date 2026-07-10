from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta
from typing import Any

import pandas as pd


HORIZON_MINUTES = {
    "5m": 5,
    "1h": 60,
    "4h": 240,
    "Daily": 1440,
}

SETTINGS_SNAPSHOT_KEYS = [
    "price_interval",
    "price_period",
    "horizon_model_lookback_bars",
    "horizon_ridge_alpha",
    "horizon_min_probability_gap",
    "filter_min_probability_gap",
    "horizon_cost_points",
    "walk_forward_tests",
    "walk_forward_min_train",
    "walk_forward_min_tests",
    "walk_forward_min_side_samples",
    "walk_forward_min_accuracy",
    "min_reward_risk",
    "min_tp1_atr_distance",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def settings_snapshot(settings: dict) -> dict:
    return {key: json_safe(settings.get(key)) for key in SETTINGS_SNAPSHOT_KEYS if key in settings}


def settings_hash(settings: dict) -> str:
    encoded = json.dumps(settings_snapshot(settings), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def scan_id(scan_time, price: float, symbol: str = "XAU/USD") -> str:
    ts = pd.Timestamp(scan_time).strftime("%Y%m%d_%H%M%S")
    raw = f"{ts}_{symbol}_{price:.2f}"
    suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    clean_symbol = symbol.replace("/", "").replace(" ", "")
    return f"{ts}_{clean_symbol}_{suffix}"


def horizon_due_time(scan_time, horizon: str) -> pd.Timestamp:
    minutes = HORIZON_MINUTES.get(str(horizon), 0)
    return pd.Timestamp(scan_time) + timedelta(minutes=minutes)


def actual_label(price_at_scan: float, price_after: float, cost_points: float) -> str:
    if price_at_scan <= 0 or price_after <= 0:
        return "invalid"
    actual_return = math.log(float(price_after) / float(price_at_scan))
    cost_return = float(cost_points) / max(float(price_at_scan), 0.01)
    if actual_return > cost_return:
        return "bullish"
    if actual_return < -cost_return:
        return "bearish"
    return "mixed"


def realized_points_for_bias(predicted_bias: str, price_at_scan: float, price_after: float) -> float:
    if predicted_bias == "bullish":
        return round(float(price_after) - float(price_at_scan), 2)
    if predicted_bias == "bearish":
        return round(float(price_at_scan) - float(price_after), 2)
    return round(abs(float(price_after) - float(price_at_scan)), 2)
