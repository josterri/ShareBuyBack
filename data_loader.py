# data_loader.py

import math
import numpy as np
import pandas as pd

def generate_gbm(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    S0: float,
    mu: float,
    sigma: float
) -> pd.DataFrame:
    """
    Simulate a Geometric Brownian Motion price path over business days.
    Returns a DataFrame with columns:
      - Date : business-day index
      - Close: simulated GBM prices
    """
    # generate business‐day index
    dates = pd.bdate_range(start=start_date, end=end_date)
    n     = len(dates)

    # convert annual parameters to daily
    mu_d    = mu / 252.0
    sigma_d = sigma / math.sqrt(252.0)

    # simulate log‐prices
    log_prices = np.empty(n, dtype=float)
    log_prices[0] = math.log(S0)

    Z = np.random.normal(size=n-1)
    increments = (mu_d - 0.5 * sigma_d**2) + sigma_d * Z
    log_prices[1:] = log_prices[0] + np.cumsum(increments)

    # back to levels
    prices = np.exp(log_prices)

    return pd.DataFrame({
        "Date":  dates,
        "Close": prices
    })
