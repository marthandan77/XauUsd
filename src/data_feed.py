from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeedResult:
    bars: pd.DataFrame
    source: str
    warning: str = ""


def make_sample_bars(rows: int = 900, freq: str = "15min") -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=rows, freq=freq)
    rng = np.random.default_rng(7)
    base = 2350 + np.cumsum(rng.normal(0, 1.45, rows))
    open_ = base + rng.normal(0, 0.55, rows)
    high = np.maximum(open_, base) + rng.uniform(0.4, 2.8, rows)
    low = np.minimum(open_, base) - rng.uniform(0.4, 2.8, rows)
    volume = rng.integers(100, 1000, rows)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": base, "volume": volume}, index=index)


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.set_index("datetime")
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required OHLC columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 1.0
    out = df[["open", "high", "low", "close", "volume"]].copy()
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    if isinstance(out.index, pd.DatetimeIndex):
        out = out[~out.index.isna()]
    return out[~out.index.duplicated(keep="last")].sort_index()


def _get_twelve_data_api_key() -> str:
    env_key = os.getenv("TWELVE_DATA_API_KEY")
    if env_key:
        return env_key.strip()
    try:
        import streamlit as st

        return str(st.secrets.get("TWELVE_DATA_API_KEY", "")).strip()
    except Exception:
        return ""


def _twelve_interval(interval: str) -> str:
    mapping = {
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }
    return mapping.get(str(interval), "15min")


def _period_days(period: str) -> int:
    period = str(period)
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("y"):
        return int(period[:-1]) * 365
    return 30


def _bars_per_day(interval: str) -> int:
    return {
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }.get(str(interval), 96)


def _outputsize(settings: dict) -> int:
    interval = str(settings.get("price_interval", "15m"))
    period = str(settings.get("price_period", "30d"))
    requested = _period_days(period) * _bars_per_day(interval)
    minimum = max(int(settings.get("minimum_bars", 500)), 1)
    maximum = max(int(settings.get("twelve_data_outputsize", 5000)), minimum)
    return max(min(requested, maximum), minimum)


def load_twelve_data_bars(settings: dict) -> FeedResult:
    api_key = _get_twelve_data_api_key()
    if not api_key:
        return FeedResult(pd.DataFrame(), "twelvedata:none", "TWELVE_DATA_API_KEY is not set")

    try:
        from twelvedata import TDClient
    except Exception as exc:
        return FeedResult(pd.DataFrame(), "twelvedata:none", f"twelvedata package unavailable: {exc}")

    symbol = str(settings.get("twelve_data_symbol", "XAU/USD"))
    interval = _twelve_interval(str(settings.get("price_interval", "15m")))
    outputsize = _outputsize(settings)
    timezone = str(settings.get("timezone", "Asia/Singapore"))

    try:
        td = TDClient(apikey=api_key)
        bars = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            timezone=timezone,
            order="asc",
        ).as_pandas()
        bars = normalize_ohlc(bars)
        if bars.empty:
            return FeedResult(pd.DataFrame(), "twelvedata:none", f"Twelve Data returned no bars for {symbol}")
        return FeedResult(bars, f"twelvedata:{symbol}", "")
    except Exception as exc:
        return FeedResult(pd.DataFrame(), "twelvedata:error", f"Twelve Data request failed: {exc}")


def load_live_bars(settings: dict) -> FeedResult:
    return load_twelve_data_bars(settings)


def load_csv(uploaded_file) -> FeedResult:
    try:
        bars = normalize_ohlc(pd.read_csv(uploaded_file))
    except Exception as exc:
        return FeedResult(pd.DataFrame(), "uploaded_csv:error", f"CSV load failed: {exc}")
    if bars.empty:
        return FeedResult(pd.DataFrame(), "uploaded_csv:error", "CSV contained no valid OHLC rows")
    return FeedResult(bars, "uploaded_csv", "")
