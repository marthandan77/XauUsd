from src.data_feed import _twelve_interval


def test_supported_twelve_data_intervals():
    assert _twelve_interval("15m") == "15min"
    assert _twelve_interval("30m") == "30min"
    assert _twelve_interval("1h") == "1h"
    assert _twelve_interval("4h") == "4h"
    assert _twelve_interval("1d") == "1day"
