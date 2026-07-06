from types import SimpleNamespace

from src.range_model import price_bands


def test_range_model_returns_bands():
    market = SimpleNamespace(price=2350.0, atr=10.0, support=2300.0, resistance=2400.0)
    out = price_bands(market, "range")
    assert out["one_hour_low"] < 2350.0
    assert out["one_hour_high"] > 2350.0
    assert out["four_hour_low"] >= 2300.0
    assert out["four_hour_high"] <= 2400.0
