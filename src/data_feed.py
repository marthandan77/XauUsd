from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests


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


def _get_massive_api_key() -> str:
    env_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if env_key:
        return env_key.strip()
    try:
        import streamlit as st

        return str(st.secrets.get("MASSIVE_API_KEY", st.secrets.get("POLYGON_API_KEY", ""))).strip()
    except Exception:
        return ""


def _massive_interval_parts(interval: str) -> tuple[int, str]:
    mapping = {
        "15m": (15, "minute"),
        "30m": (30, "minute"),
        "1h": (1, "hour"),
        "4h": (4, "hour"),
        "1d": (1, "day"),
    }
    return mapping.get(str(interval), (15, "minute"))


def _period_to_dates(period: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    period = str(period)
    if period.endswith("d"):
        days = int(period[:-1])
    elif period.endswith("mo"):
        days = int(period[:-2]) * 30
    elif period.endswith("y"):
        days = int(period[:-1]) * 365
    else:
        days = 30
    start = now - timedelta(days=days)
    return start.date().isoformat(), now.date().isoformat()


def _massive_symbols(settings: dict) -> list[str]:
    primary = str(settings.get("massive_symbol_primary", "C:XAUUSD"))
    fallback = str(settings.get("massive_symbol_fallback", "XAUUSD"))
    candidates = [primary, fallback, "C:XAUUSD", "XAUUSD"]
    extra = settings.get("massive_symbol_extra_fallbacks", []) or []
    if isinstance(extra, str):
        candidates.extend([x.strip() for x in extra.split(",") if x.strip()])
    else:
        candidates.extend([str(x) for x in extra])
    clean: list[str] = []
    for symbol in candidates:
        if symbol and symbol not in clean:
            clean.append(symbol)
    return clean


def _massive_results_to_bars(results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in results or []:
        try:
            rows.append(
                {
                    "time": pd.to_datetime(int(item["t"]), unit="ms", utc=True),
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item.get("v", item.get("n", 1.0)) or 1.0),
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return normalize_ohlc(df)


def load_massive_bars(settings: dict) -> FeedResult:
    api_key = _get_massive_api_key()
    if not api_key:
        return FeedResult(pd.DataFrame(), "massive:none", "MASSIVE_API_KEY is not set")

    interval = str(settings.get("price_interval", "15m"))
    multiplier, timespan = _massive_interval_parts(interval)
    from_date, to_date = _period_to_dates(str(settings.get("price_period", "30d")))
    base_url = str(settings.get("massive_base_url", "https://api.massive.com")).rstrip("/")

    warnings = []
    for symbol in _massive_symbols(settings):
        url = f"{base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": int(settings.get("massive_limit", 50000))}
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.get(url, params=params, headers=headers, timeout=int(settings.get("massive_timeout", 20)))
            if response.status_code in {401, 403}:
                return FeedResult(pd.DataFrame(), "massive:auth_error", "Massive API authorization failed. Check MASSIVE_API_KEY and plan access.")
            if response.status_code == 429:
                return FeedResult(pd.DataFrame(), "massive:rate_limited", "Massive API rate limit hit. Wait or reduce refresh frequency.")
            response.raise_for_status()
            payload = response.json()
            bars = _massive_results_to_bars(payload.get("results", []))
            if not bars.empty:
                return FeedResult(bars, f"massive:{symbol}", "")
            warnings.append(f"{symbol}: empty results ({payload.get('status', 'unknown status')})")
        except Exception as exc:
            warnings.append(f"{symbol}: {exc}")

    return FeedResult(pd.DataFrame(), "massive:none", "; ".join(warnings) or "no Massive bars returned")


def _symbol_candidates(settings: dict) -> list[str]:
    candidates = [
        str(settings.get("symbol_primary", "XAUUSD=X")),
        str(settings.get("symbol_fallback", "GC=F")),
        "GC=F",
        "MGC=F",
    ]
    extra = settings.get("symbol_extra_fallbacks", []) or []
    if isinstance(extra, str):
        candidates.extend([x.strip() for x in extra.split(",") if x.strip()])
    else:
        candidates.extend([str(x) for x in extra])
    clean: list[str] = []
    for symbol in candidates:
        if symbol and symbol not in clean:
            clean.append(symbol)
    return clean


def _safe_period(interval: str, requested_period: str) -> str:
    # Yahoo intraday feeds are more reliable with bounded periods.
    if interval in {"15m", "30m", "1h"} and requested_period in {"6mo", "1y", "2y", "5y", "max"}:
        return "60d"
    return requested_period


def load_yfinance_bars(settings: dict) -> FeedResult:
    requested_interval = str(settings.get("price_interval", "15m"))
    yfinance_interval = {"4h": "1h"}.get(requested_interval, requested_interval)
    period = _safe_period(yfinance_interval, str(settings.get("price_period", "30d")))
    try:
        import yfinance as yf
    except Exception as exc:
        return FeedResult(pd.DataFrame(), "yfinance:none", f"yfinance unavailable: {exc}")

    warnings = []
    for symbol in _symbol_candidates(settings):
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval=yfinance_interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            bars = normalize_ohlc(raw)
            if requested_interval == "4h" and not bars.empty:
                bars = _resample_ohlc(bars, "4h")
            if not bars.empty:
                warn = "" if requested_interval != "4h" else "4h bars resampled from 1h feed"
                if symbol != str(settings.get("symbol_primary", "XAUUSD=X")):
                    warn = (warn + "; " if warn else "") + f"primary feed failed; using fallback {symbol}"
                return FeedResult(bars, f"yfinance:{symbol}", warn)
            warnings.append(f"{symbol}: empty response")
        except Exception as exc:
            warnings.append(f"{symbol}: {exc}")
    return FeedResult(pd.DataFrame(), "yfinance:none", "; ".join(warnings) or "no yfinance bars returned")


def load_live_bars(settings: dict) -> FeedResult:
    provider = str(settings.get("data_provider", "massive")).lower()

    if provider == "massive":
        feed = load_massive_bars(settings)
        if not feed.bars.empty:
            return feed
        if bool(settings.get("fallback_to_yfinance", True)):
            fallback = load_yfinance_bars(settings)
            warning = feed.warning
            if not fallback.bars.empty:
                fallback.warning = f"Massive unavailable: {warning}; using {fallback.source}"
                return fallback
        return feed

    if provider == "yfinance":
        return load_yfinance_bars(settings)

    return FeedResult(pd.DataFrame(), "none", f"unknown data_provider: {provider}")


def load_csv(uploaded_file) -> FeedResult:
    bars = normalize_ohlc(pd.read_csv(uploaded_file))
    return FeedResult(bars, "uploaded_csv", "")
