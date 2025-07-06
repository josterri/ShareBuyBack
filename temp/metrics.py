import pandas as pd
import numpy as np

def calculate_vwap(prices: pd.Series, volumes: pd.Series) -> float:
    """
    Calculate Volume-Weighted Average Price (VWAP) over the given price and volume series.
    VWAP = sum(price_i * volume_i) / sum(volume_i)
    """
    if len(prices) != len(volumes) or len(prices) == 0:
        raise ValueError("Price and volume series must be the same length and non-empty.")
    total_vol = volumes.sum()
    if total_vol == 0:
        return 0.0
    vwap_value = (prices * volumes).sum() / total_vol
    return float(vwap_value)

def calculate_twap(prices: pd.Series) -> float:
    """
    Calculate Time-Weighted Average Price (TWAP) over the given price series.
    TWAP = arithmetic average of prices over time.
    """
    if len(prices) == 0:
        return 0.0
    twap_value = prices.mean()
    return float(twap_value)

def calculate_harmonic_mean(prices: pd.Series) -> float:
    """
    Calculate the harmonic mean of the given price series.
    Harmonic mean = n / sum(1/price_i) for i from 1 to n.
    """
    n = len(prices)
    if n == 0:
        return 0.0
    # Replace any zero prices to avoid division by zero (though stock prices won't be zero normally)
    prices_nonzero = prices.replace(0, np.nan).dropna()
    n_nonzero = len(prices_nonzero)
    if n_nonzero == 0:
        return 0.0
    harmonic_mean_value = n_nonzero / (1.0 / prices_nonzero).sum()
    return float(harmonic_mean_value)
