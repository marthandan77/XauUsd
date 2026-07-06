from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(bars: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = bars.copy()
    if df.empty:
        return df
    ema_fast = int(settings.get("ema_fast", 50))
    ema_slow = int(settings.get("ema_slow", 200))
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].replace(0, 1).fillna(1)
    df["vwap"] = (typical * volume).cumsum() / volume.cumsum()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr"] = tr.rolling(14, min_periods=3).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean().replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + gain / loss))).fillna(50)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = df["atr"].replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=3).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=3).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.rolling(14, min_periods=3).mean().fillna(0).clip(0, 100)
    return df
