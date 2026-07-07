from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCAN_LOG_PATH = ROOT / "data" / "scan_log.csv"


def append_scan_log(row: dict) -> Path:
    SCAN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in row.items() if isinstance(key, str)}
    frame = pd.DataFrame([clean])
    header = not SCAN_LOG_PATH.exists()
    frame.to_csv(SCAN_LOG_PATH, mode="a", header=header, index=False)
    return SCAN_LOG_PATH


def read_scan_log(limit: int = 200) -> pd.DataFrame:
    if not SCAN_LOG_PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(SCAN_LOG_PATH)
    if limit and len(data) > limit:
        return data.tail(int(limit)).copy()
    return data
