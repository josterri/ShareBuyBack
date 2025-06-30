import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical daily 'Close' and 'Volume' data from Yahoo Finance.

    Parameters:
        ticker (str): e.g. 'AAPL'
        start_date (str): e.g. '2023-01-01' or '2023/01/01'
        end_date (str): e.g. '2023-12-31' or '2023/12/31'

    Returns:
        pd.DataFrame: with columns ['Date', 'Close', 'Volume']
    """
    # Normalize date separators
    start_str = start_date.replace('/', '-')
    end_str = end_date.replace('/', '-')
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    except ValueError:
        raise RuntimeError(f"Invalid date format: {start_date} or {end_date}")

    # Adjust end date if in future
    today = datetime.now()
    if end_dt >= today:
        end_dt = today - timedelta(days=1)

    # Primary download
    df = yf.download(ticker, start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
    if df.empty:
        # Fallback: use history and then filter
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))
        if hist.empty:
            raise RuntimeError(f"No data for {ticker} in range {start_dt.date()} to {end_dt.date()}")
        df = hist

    # Clean and format
    df = df[['Close', 'Volume']].dropna()
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df[['Date', 'Close', 'Volume']]
