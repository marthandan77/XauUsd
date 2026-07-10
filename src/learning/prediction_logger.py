from __future__ import annotations

from datetime import date

from .github_store import GitHubLearningStore
from .schema import json_safe, scan_id, settings_hash, settings_snapshot


HORIZON_EVENT_KEYS = [
    "horizon",
    "role",
    "bias",
    "bull_confidence",
    "bear_confidence",
    "validation_status",
    "validation_tests",
    "bull_validation_samples",
    "bear_validation_samples",
    "active_accuracy",
    "active_samples",
    "mae_points",
    "decision",
    "ev_buy_points",
    "ev_sell_points",
    "min_ev_points",
    "model_up_probability",
    "model_down_probability",
    "expected_low",
    "expected_high",
]


def _horizon_event(item: dict) -> dict:
    return {key: json_safe(item.get(key)) for key in HORIZON_EVENT_KEYS if key in item}


def build_prediction_event(result: dict) -> dict:
    settings = result.get("settings", {})
    market = result.get("market")
    price = float(getattr(market, "price", 0.0) or 0.0)
    symbol = str(settings.get("twelve_data_symbol", "XAU/USD"))
    scan_time = result.get("scan_time")
    event_id = scan_id(scan_time, price, symbol=symbol)

    return {
        "schema_version": 1,
        "scan_id": event_id,
        "scan_time": json_safe(scan_time),
        "symbol": symbol,
        "timeframe": settings.get("price_interval"),
        "price_at_scan": price,
        "feed_source": getattr(result.get("feed"), "source", "unknown"),
        "raw_action": result.get("action"),
        "final_action": result.get("veto", {}).get("final_action"),
        "veto_reasons": json_safe(result.get("veto", {}).get("reasons", [])),
        "horizons": [_horizon_event(item) for item in result.get("multi_horizon", [])],
        "settings_hash": settings_hash(settings),
        "settings_used": settings_snapshot(settings),
    }


def prediction_path(event: dict) -> str:
    ts = str(event.get("scan_time", ""))[:10] or date.today().isoformat()
    return f"learning/predictions/{ts}/{event['scan_id']}.json"


def log_prediction(result: dict, store: GitHubLearningStore | None = None) -> dict:
    store = store or GitHubLearningStore.from_runtime()
    if not store.enabled:
        return {"ok": False, "stage": "log_prediction", "reason": "GitHub learning store is not configured", "store": store.status()}
    if not result.get("ok"):
        return {"ok": False, "stage": "log_prediction", "reason": "scan result is not ok"}
    event = build_prediction_event(result)
    path = prediction_path(event)
    write_result = store.write_json(path, event, f"learning: log prediction {event['scan_id']}")
    return {"ok": bool(write_result.get("ok")), "stage": "log_prediction", "path": path, "scan_id": event["scan_id"], "write": write_result}
