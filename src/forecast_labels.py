from __future__ import annotations

import pandas as pd

HORIZONS: dict[str, int] = {
    "30m": 2,
    "1h": 4,
    "4h": 16,
    "8h": 32,
}


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
    horizons = horizons or HORIZONS
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
