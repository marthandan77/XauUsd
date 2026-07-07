from __future__ import annotations

import pandas as pd

TARGET_HORIZON_MINUTES: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "Day": 1440,
}

INTERVAL_MINUTES: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def interval_minutes(interval: str) -> int:
    return INTERVAL_MINUTES.get(str(interval), 15)


def horizons_for_interval(interval: str) -> dict[str, int]:
    """Return forecast horizons that can be measured exactly from the selected candle interval.

    Example:
    - 5m chart: 5m=1 candle, 15m=3 candles, 1h=12 candles, 4h=48 candles, Day=288 candles.
    - 15m chart: 15m=1 candle, 1h=4 candles, 4h=16 candles, Day=96 candles.

    A shorter horizon than the selected candle size is not returned because it cannot be
    calculated from actual candles without inventing intrabar data.
    """
    source_minutes = interval_minutes(interval)
    horizons: dict[str, int] = {}
    for label, target_minutes in TARGET_HORIZON_MINUTES.items():
        if target_minutes < source_minutes:
            continue
        if target_minutes % source_minutes != 0:
            continue
        horizons[label] = target_minutes // source_minutes
    return horizons


def _future_high(df: pd.DataFrame, bars_ahead: int) -> pd.Series:
    future_highs = [df["high"].shift(-i) for i in range(1, int(bars_ahead) + 1)]
    return pd.concat(future_highs, axis=1).max(axis=1, skipna=False)


def _future_low(df: pd.DataFrame, bars_ahead: int) -> pd.Series:
    future_lows = [df["low"].shift(-i) for i in range(1, int(bars_ahead) + 1)]
    return pd.concat(future_lows, axis=1).min(axis=1, skipna=False)


def add_forward_outcomes(df: pd.DataFrame, horizons: dict[str, int] | None = None) -> pd.DataFrame:
    """Add actual future outcome labels for each forecast horizon.

    These columns are calculated only from historical candles that already exist in the
    loaded dataframe. No projected ATR, drift, or synthetic price path is created.
    """
    horizons = horizons or horizons_for_interval("15m")
    out = df.copy()
    if out.empty:
        return out

    close = out["close"].astype(float)
    for label, bars_ahead in horizons.items():
        future_high = _future_high(out, bars_ahead).astype(float)
        future_low = _future_low(out, bars_ahead).astype(float)
        future_close = close.shift(-int(bars_ahead)).astype(float)

        out[f"{label}_future_high"] = future_high
        out[f"{label}_future_low"] = future_low
        out[f"{label}_future_close"] = future_close
        out[f"{label}_future_close_change"] = future_close - close

        out[f"{label}_buy_profit"] = future_high - close
        out[f"{label}_buy_adverse"] = close - future_low
        out[f"{label}_sell_profit"] = close - future_low
        out[f"{label}_sell_adverse"] = future_high - close

    return out
