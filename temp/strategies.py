# strategies.py
import pandas as pd

def simulate_twap(data: pd.DataFrame, total_shares: float) -> pd.DataFrame:
    days = len(data)
    per_day = total_shares / days
    records = []
    for _, row in data.iterrows():
        shares = per_day
        cost = shares * row['Close']
        records.append({
            'Date': row['Date'],
            'Price': row['Close'],
            'Shares': shares,
            'Cost': cost
        })
    return pd.DataFrame(records)

def simulate_volume_participation(
    data: pd.DataFrame,
    participation_frac: float
) -> pd.DataFrame:
    records = []
    for _, row in data.iterrows():
        shares = row['Volume'] * participation_frac
        cost = shares * row['Close']
        records.append({
            'Date': row['Date'],
            'Price': row['Close'],
            'Shares': shares,
            'Cost': cost
        })
    return pd.DataFrame(records)
