# benchmarks.py
import numpy as np
import pandas as pd

def vwap(prices: pd.Series, volumes: pd.Series) -> float:
    total_vol = volumes.sum()
    return float((prices * volumes).sum() / total_vol) if total_vol>0 else np.nan

def twap(prices: pd.Series) -> float:
    return float(prices.mean())

def harmonic_mean(prices: pd.Series) -> float:
    arr = prices.replace(0, np.nan).dropna()
    return float(len(arr) / (1.0/arr).sum()) if len(arr)>0 else np.nan
