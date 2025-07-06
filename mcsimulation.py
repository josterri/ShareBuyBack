# mcsimulation.py

import numpy as np
import pandas as pd
from diskcache import Cache

CACHE = Cache("./mc_cache_disk")  # creates one .db file + folder

def _make_key(S0, mu, sigma, horizon, sims, total_shares):
    # use a tuple so lookup is exact
    return (S0, mu, sigma, horizon, sims, total_shares)

def load_or_run_simulation(
    S0: float,
    mu: float,
    sigma: float,
    horizon: int,
    sims: int,
    total_shares: float = 100.0
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ["P_twap","P_usd"] of length `sims`,
    loading from diskcache if available.
    """
    key = _make_key(S0, mu, sigma, horizon, sims, total_shares)
    df = CACHE.get(key)
    if df is not None:
        return df

    # simulate GBM log‐returns
    mu_d    = mu    / 252.0
    sigma_d = sigma / np.sqrt(252.0)
    Z       = np.random.normal(size=(horizon, sims))
    logs    = np.vstack([
        np.zeros(sims),
        np.cumsum((mu_d - 0.5*sigma_d**2) + sigma_d * Z, axis=0)
    ])
    price_paths = S0 * np.exp(logs)  # shape (horizon+1, sims)

    # TWAP
    shares_per_day = total_shares / horizon
    twap_cost      = (price_paths[1:] * shares_per_day).sum(axis=0)
    P_twap         = twap_cost / total_shares

    # Fixed‐Notional
    daily_notional = total_shares * S0 / horizon
    shares_usd     = (daily_notional / price_paths[1:]).sum(axis=0)
    P_usd          = (daily_notional * horizon) / shares_usd

    df = pd.DataFrame({"P_twap": P_twap, "P_usd": P_usd})
    CACHE.set(key, df)  # store the full DataFrame
    return df
