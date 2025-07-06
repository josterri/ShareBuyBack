import pytest
import pandas as pd
from io import StringIO
from core.utils import load_price_data

def test_valid_csv():
    csv_data = StringIO("Date,Close,Volume\n2023-01-01,100,1000\n2023-01-02,101,1100")
    df = load_price_data(csv_data)
    assert not df.empty
    assert list(df.columns) == ["Date", "Close", "Volume"]

def test_missing_required_column():
    csv_data = StringIO("Date,Price\n2023-01-01,100\n2023-01-02,101")
    with pytest.raises(RuntimeError):
        load_price_data(csv_data)

def test_no_date_column():
    csv_data = StringIO("Close,Volume\n100,1000\n101,1100")
    df = load_price_data(csv_data)
    assert "Date" in df.columns
    assert len(df["Date"]) == 2
