import numpy as np
import pandas as pd

def vwap(prices: pd.Series, volumes: pd.Series) -> float:
    """
    Volume-Weighted Average Price

    Parameters:
        prices (pd.Series): Series of prices
        volumes (pd.Series): Series of volumes

    Returns:
        float: VWAP value
    """
    return np.average(prices, weights=volumes)

def twap(prices: pd.Series) -> float:
    """
    Time-Weighted Average Price

    Parameters:
        prices (pd.Series): Series of prices

    Returns:
        float: TWAP value
    """
    return prices.mean()

def harmonic_mean(prices: pd.Series) -> float:
    """
    Harmonic mean of prices

    Parameters:
        prices (pd.Series): Series of prices

    Returns:
        float: Harmonic mean value
    """
    return len(prices) / np.sum(1.0 / prices)
