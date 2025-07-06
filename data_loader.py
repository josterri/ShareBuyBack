# data_loader.py
import math
import numpy as np
import pandas as pd
import yfinance as yf

def generate_gbm(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    S0: float,
    mu: float,
    sigma: float,
    avg_vol: int
) -> pd.DataFrame:
    # business days between
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    mu_d = mu/252.0
    sigma_d = sigma/math.sqrt(252.0)
    prices = [S0]
    for _ in range(n-1):
        z = np.random.normal()
        prices.append(prices[-1] * math.exp((mu_d - 0.5*sigma_d**2) + sigma_d*z))
    vols = np.full(n, avg_vol, dtype=float)
    vols *= np.exp(np.random.normal(0, 0.1, size=n))
    vols = vols.round().astype(int)
    return pd.DataFrame({'Date': dates, 'Close': prices, 'Volume': vols})
