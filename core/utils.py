import pandas as pd
import io

def load_price_data(file) -> pd.DataFrame:
    """
    Loads price and volume data from a CSV file.

    Parameters:
        file: Streamlit file buffer or file path

    Returns:
        pd.DataFrame: with columns ['Date', 'Close', 'Volume']
    """
    try:
        if hasattr(file, 'read'):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(open(file, "r"))

        required = {'Close', 'Volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required}")

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        else:
            df['Date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))

        return df[['Date', 'Close', 'Volume']].dropna()

    except Exception as e:
        raise RuntimeError(f"Failed to load price data: {e}")
