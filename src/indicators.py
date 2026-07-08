from __future__ import annotations

import numpy as np
import pandas as pd


def _to_int(settings: dict, key: str, default: int) -> int:
    return max(int(settings.get(key, default)), 1)


def _to_float(settings: dict, key: str, default: float) -> float:
    return float(settings.get(key, default))


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(max(int(length), 1), min_periods=max(min(int(length), 5), 1)).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = true_range(df)
    return tr.rolling(max(int(length), 1), min_periods=max(min(int(length), 5), 1)).mean()


def bollinger_bands(series: pd.Series, length: int, multiplier: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(series, length)
    std = series.rolling(max(int(length), 1), min_periods=max(min(int(length), 5), 1)).std(ddof=0)
    upper = middle + float(multiplier) * std
    lower = middle - float(multiplier) * std
    return lower, middle, upper


def keltner_channels(df: pd.DataFrame, length: int, multiplier: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(df["close"], length)
    range_ma = true_range(df).rolling(max(int(length), 1), min_periods=max(min(int(length), 5), 1)).mean()
    upper = middle + float(multiplier) * range_ma
    lower = middle - float(multiplier) * range_ma
    return lower, middle, upper


def realized_volatility(series: pd.Series, length: int, bars_per_year: int = 252) -> pd.Series:
    returns = np.log(series / series.shift(1))
    return returns.rolling(max(int(length), 2), min_periods=max(min(int(length), 10), 2)).std() * np.sqrt(bars_per_year)


def _linear_regression_last_value(values: np.ndarray) -> float:
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values.astype(float), 1)
    return float(slope * (len(values) - 1) + intercept)


def momentum_value(df: pd.DataFrame, length: int) -> pd.Series:
    length = max(int(length), 2)
    highest_high = df["high"].rolling(length, min_periods=length).max()
    lowest_low = df["low"].rolling(length, min_periods=length).min()
    sma_close = sma(df["close"], length)
    midline = ((highest_high + lowest_low) / 2.0 + sma_close) / 2.0
    momentum_source = df["close"] - midline
    return momentum_source.rolling(length, min_periods=length).apply(_linear_regression_last_value, raw=True)


def _bars_since_true(mask: pd.Series) -> pd.Series:
    """Return number of bars since the last True value; NaN if no True has occurred."""
    bar_number = pd.Series(np.arange(len(mask)), index=mask.index, dtype=float)
    last_true_bar = bar_number.where(mask.fillna(False).astype(bool)).ffill()
    return bar_number - last_true_bar


def add_indicators(bars: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = bars.copy()
    if df.empty:
        return df

    ema_fast = _to_int(settings, "ema_fast", 50)
    ema_slow = _to_int(settings, "ema_slow", 200)
    atr_period = _to_int(settings, "atr_period", 14)
    trend_length = _to_int(settings, "trend_length", ema_slow)
    bb_length = _to_int(settings, "bb_length", 20)
    bb_mult = _to_float(settings, "bb_mult", 2.0)
    kc_length = _to_int(settings, "kc_length", 20)
    kc_mult = _to_float(settings, "kc_mult", 1.5)
    rv_length = _to_int(settings, "realized_vol_length", 100)
    release_recent_bars = _to_int(settings, "kc_release_recent_bars", 3)
    chase_atr_limit = _to_float(settings, "kc_release_chase_atr_limit", 1.0)

    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()
    df["trend_sma"] = sma(df["close"], trend_length)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].replace(0, 1).fillna(1)
    df["vwap"] = (typical * volume).cumsum() / volume.cumsum()

    df["tr"] = true_range(df)
    df["atr"] = atr(df, atr_period)

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=3).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=3).mean().replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + gain / loss))).fillna(50)

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_series = df["atr"].replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=3).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=3).mean() / atr_series
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"] = dx.rolling(14, min_periods=3).mean().fillna(0).clip(0, 100)

    df["bb_lower"], df["bb_middle"], df["bb_upper"] = bollinger_bands(df["close"], bb_length, bb_mult)
    df["kc_lower"], df["kc_middle"], df["kc_upper"] = keltner_channels(df, kc_length, kc_mult)

    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]).abs()
    df["kc_width"] = (df["kc_upper"] - df["kc_lower"]).abs()
    df["compression_ratio"] = df["bb_width"] / df["kc_width"].replace(0, np.nan)

    df["squeeze_on"] = (df["bb_lower"] >= df["kc_lower"]) & (df["bb_upper"] <= df["kc_upper"])
    df["squeeze_fired"] = df["squeeze_on"].shift(1).fillna(False).astype(bool) & (~df["squeeze_on"].fillna(False).astype(bool))

    df["bars_since_squeeze_release"] = _bars_since_true(df["squeeze_fired"])
    df["squeeze_recent"] = (
        df["bars_since_squeeze_release"].notna()
        & (df["bars_since_squeeze_release"] >= 0)
        & (df["bars_since_squeeze_release"] <= release_recent_bars)
    )

    df["release_close"] = df["close"].where(df["squeeze_fired"]).ffill()
    df["release_high"] = df["high"].where(df["squeeze_fired"]).ffill()
    df["release_low"] = df["low"].where(df["squeeze_fired"]).ffill()
    df["release_chase_atr"] = ((df["close"] - df["release_close"]).abs() / df["atr"].replace(0, np.nan)).where(df["squeeze_recent"])
    df["release_chase_risk"] = df["squeeze_recent"] & (df["release_chase_atr"] > chase_atr_limit)

    df["kc_momentum"] = momentum_value(df, kc_length)
    df["kc_momentum_prev"] = df["kc_momentum"].shift(1)
    df["kc_momentum_rising"] = df["kc_momentum"] > df["kc_momentum_prev"]
    df["kc_momentum_falling"] = df["kc_momentum"] < df["kc_momentum_prev"]

    df["release_bullish_confirmed"] = (
        df["squeeze_recent"]
        & (df["close"] > df["release_high"])
        & (df["kc_momentum"] > 0)
        & (df["kc_momentum"] >= df["kc_momentum_prev"])
    )
    df["release_bearish_confirmed"] = (
        df["squeeze_recent"]
        & (df["close"] < df["release_low"])
        & (df["kc_momentum"] < 0)
        & (df["kc_momentum"] <= df["kc_momentum_prev"])
    )

    df["release_direction"] = "none"
    df.loc[df["squeeze_recent"] & ~(df["release_bullish_confirmed"] | df["release_bearish_confirmed"]), "release_direction"] = "unconfirmed"
    df.loc[df["release_bullish_confirmed"], "release_direction"] = "bullish"
    df.loc[df["release_bearish_confirmed"], "release_direction"] = "bearish"

    df["realized_vol"] = realized_volatility(df["close"], rv_length)

    return df
