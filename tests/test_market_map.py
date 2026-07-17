import pandas as pd

from src.indicators import add_indicators
from src.market_map import build_market_map


def _bars(rows: int, freq: str) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq=freq)
    close = pd.Series(range(rows), index=index, dtype=float) + 2300.0
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 100.0,
        },
        index=index,
    )


def test_market_map_uses_timeframe_specific_seven_day_window():
    settings = {
        "price_interval": "1h",
        "lookback_days": 7,
        "atr_period": 14,
        "ema_fast": 20,
        "ema_slow": 50,
        "trend_length": 50,
    }
    enriched = add_indicators(_bars(500, "1h"), settings).dropna(subset=["atr", "trend_sma"])
    market = build_market_map(enriched, settings)

    expected_window = enriched.tail(7 * 24)
    assert market.high_7d == float(expected_window["high"].max())
    assert market.low_7d == float(expected_window["low"].min())


def test_daily_market_map_uses_seven_bars_for_seven_days():
    settings = {
        "price_interval": "1d",
        "lookback_days": 7,
        "atr_period": 5,
        "ema_fast": 5,
        "ema_slow": 10,
        "trend_length": 10,
    }
    enriched = add_indicators(_bars(50, "1D"), settings).dropna(subset=["atr", "trend_sma"])
    market = build_market_map(enriched, settings)

    expected_window = enriched.tail(7)
    assert market.high_7d == float(expected_window["high"].max())
    assert market.low_7d == float(expected_window["low"].min())
