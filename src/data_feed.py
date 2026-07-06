from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeedResult:
    bars: pd.DataFrame
    source: str
    warning: str = ""


def make_sample_bars(rows: int = 640, freq: str = "15min") -> pd.DataFrame:
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required OHLC columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 1.0
    out = df[["open", "high", "low", "close", "volume"]].copy()
    return out.dropna(subset=["open", "high", "low", "close"])


def _resample_ohlc(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    if bars.empty:
        return bars
    return (
        bars.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )


def load_live_bars(settings: dict) -> FeedResult:
    symbols = [settings.get("symbol_primary", "XAUUSD=X"), settings.get("symbol_fallback", "GC=F")]
    period = settings.get("price_period", "30d")
    requested_interval = str(settings.get("price_interval", "15m"))
    yfinance_interval = {"4h": "1h"}.get(requested_interval, requested_interval)
    try:
        import yfinance as yf
    except Exception as exc:
        return FeedResult(pd.DataFrame(), "none", f"yfinance unavailable: {exc}")
    warnings = []
    for symbol in symbols:
        try:
            raw = yf.download(symbol, period=period, interval=yfinance_interval, progress=False, auto_adjust=False)
            bars = normalize_ohlc(raw)
            if requested_interval == "4h" and not bars.empty:
                bars = _resample_ohlc(bars, "4h")
            if not bars.empty:
                return FeedResult(bars, symbol, "" if requested_interval != "4h" else "4h bars resampled from 1h feed")
        except Exception as exc:
            warnings.append(f"{symbol}: {exc}")
    return FeedResult(pd.DataFrame(), "none", "; ".join(warnings) or "no live bars returned")


def load_csv(uploaded_file) -> FeedResult:
    bars = normalize_ohlc(pd.read_csv(uploaded_file))
    return FeedResult(bars, "uploaded_csv", "")
