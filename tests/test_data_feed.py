from io import StringIO

from src.data_feed import load_csv


def test_valid_csv_is_normalized():
    uploaded = StringIO("time,open,high,low,close\n2026-01-01,1,2,0.5,1.5\n")
    result = load_csv(uploaded)

    assert result.source == "uploaded_csv"
    assert result.warning == ""
    assert list(result.bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result.bars) == 1


def test_invalid_csv_returns_controlled_error():
    uploaded = StringIO("time,price\n2026-01-01,1.5\n")
    result = load_csv(uploaded)

    assert result.bars.empty
    assert result.source == "uploaded_csv:error"
    assert "missing required OHLC columns" in result.warning
