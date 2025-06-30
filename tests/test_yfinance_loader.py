import pytest
from live.yfinance_loader import download_data

def test_download_data_valid():
    df = download_data("AAPL", "2023-01-01", "2023-01-10")
    assert not df.empty
    assert set(["Date", "Close", "Volume"]).issubset(df.columns)

def test_download_data_invalid_ticker():
    with pytest.raises(RuntimeError):
        download_data("INVALID_TICKER_ABC123", "2023-01-01", "2023-01-10")
