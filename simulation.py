# simulation.py
import math
import numpy as np
import pandas as pd
from strategies import simulate_twap, simulate_volume_participation

def run_monte_carlo(
    S0: float,
    avg_vol: float,
    mu: float,
    sigma: float,
    horizon: int,
    sims: int,
    total_shares: float,
    participation_frac: float
) -> pd.DataFrame:
    # daily parameters
    mu_d = mu/252.0
    sigma_d = sigma/math.sqrt(252.0)
    costs = []
    for _ in range(sims):
        # generate price path
        zs = np.random.normal(size=horizon)
        logrets = (mu_d - 0.5*sigma_d**2) + sigma_d*zs
        cum = np.cumsum(logrets)
        prices = S0 * np.exp(cum)
        # build a temp DataFrame
        df = pd.DataFrame({
            'Date': range(horizon),
            'Close': prices,
            'Volume': np.full(horizon, avg_vol)
        })
        twap_df = simulate_twap(df, total_shares)
        vp_df = simulate_volume_participation(df, participation_frac)
        cost_twap = twap_df['Cost'].sum()
        cost_vp = vp_df['Cost'].sum()
        costs.append({'TWAP': cost_twap, 'VP': cost_vp})
    return pd.DataFrame(costs)
