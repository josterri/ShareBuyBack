# monte_carlo_tab.py

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from simulation import run_monte_carlo

# Cache directory next to this file
CACHE_DIR = os.path.join(os.path.dirname(__file__), "mc_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_path(
    initial_price: float,
    avg_vol: float,
    mu: float,
    sigma: float,
    horizon: int,
    sims: int,
    total_shares: float
) -> str:
    """
    Build a human-readable filename based on all user inputs plus avg_vol.
    """
    fname = (
        f"mc_S0_{initial_price:.2f}"
        f"_vol_{avg_vol:.0f}"
        f"_mu_{mu:.4f}"
        f"_sigma_{sigma:.4f}"
        f"_h{horizon}"
        f"_s{sims}"
        f"_n{int(total_shares)}"
        ".csv"
    )
    return os.path.join(CACHE_DIR, fname)


def load_or_run_simulation(
    initial_price: float,
    avg_vol: float,
    mu: float,
    sigma: float,
    horizon: int,
    sims: int,
    total_shares: float
) -> pd.DataFrame:
    """
    Look for a cached CSV of the exact same parameters (including avg_vol),
    otherwise run the sim and write it out.
    """
    cache_path = _get_cache_path(
        initial_price, avg_vol, mu, sigma, horizon, sims, total_shares
    )
    st.write(f"🔍 Looking for cache at `{cache_path}`")

    if os.path.exists(cache_path):
        st.success(f"✅ Loaded cached MC results ({os.path.basename(cache_path)})")
        return pd.read_csv(cache_path)

    st.warning("🔄 No cache hit; running Monte Carlo…")
    df = run_monte_carlo(
        S0=initial_price,
        avg_vol=avg_vol,
        mu=mu,
        sigma=sigma,
        horizon=horizon,
        sims=sims,
        total_shares=total_shares
    )
    df.to_csv(cache_path, index=False)
    st.success(f"💾 Saved new results to cache ({os.path.basename(cache_path)})")
    return df


# monte_carlo_tab.py

# ... keep your imports and CACHE_DIR stuff unchanged ...

def run_monte_carlo_tab(
    df: pd.DataFrame,
    initial_price: float,
    avg_vol: float,           # <-- new parameter
    mc_drift: float,
    mc_vol: float,
    mc_horiz: int,
    mc_sims: int,
    total_shares: float
):
    """
    Monte Carlo Simulation Analysis:
      1. Price‐based outperformance (bps)
      2. Simulated GBM price paths
    """
    # no longer pulling avg_vol from df:
    # avg_vol = df["Volume"].mean()

    # 1) load or run using the user‐supplied avg_vol
    costs_df = load_or_run_simulation(
        initial_price,
        avg_vol,
        mc_drift,
        mc_vol,
        mc_horiz,
        mc_sims,
        total_shares
    )

    # 2) Build future GBM paths and compute bps outperformance
    mu_d = mc_drift / 252.0
    sigma_d = mc_vol / np.sqrt(252.0)
    Z = np.random.normal(size=(mc_horiz, mc_sims))
    logs = np.vstack([
        np.zeros(mc_sims),
        np.cumsum((mu_d - 0.5 * sigma_d**2) + sigma_d * Z, axis=0)
    ])
    price_paths = initial_price * np.exp(logs)

    daily_notional = total_shares * initial_price / mc_horiz
    shares_usd     = (daily_notional / price_paths[1:]).sum(axis=0)
    fixed_avg      = price_paths[1:].mean(axis=0)
    usd_avg        = (total_shares * initial_price) / shares_usd
    bps_diff       = (fixed_avg - usd_avg) / fixed_avg * 1e4

    st.subheader("1. Price Difference in Basis Points")
    df_bps = pd.DataFrame({'Price Outperformance (bps)': bps_diff})
    fig_bps = px.histogram(
        df_bps,
        x='Price Outperformance (bps)',
        nbins=200,
        title='Distribution of Price Outperformance (bps)'
    )
    fig_bps.update_layout(xaxis_title='bps Δ', yaxis_title='Frequency')
    st.plotly_chart(fig_bps, use_container_width=True)
    st.metric("Mean bps Δ", f"{bps_diff.mean():.1f}", f"σ={bps_diff.std():.1f}")

    st.subheader("2. Simulated GBM Price Paths")
    max_paths = min(price_paths.shape[1], 100)
    melt = (
        pd.DataFrame(price_paths[:, :max_paths])
          .reset_index()
          .melt(id_vars='index', var_name='sim', value_name='price')
          .rename(columns={'index': 'Day'})
    )
    fig_paths = px.line(
        melt,
        x='Day', y='price', color='sim', line_group='sim',
        title=f"Simulated GBM Paths (showing {max_paths} of {mc_sims} sims)"
    )
    fig_paths.update_layout(showlegend=False)
    st.plotly_chart(fig_paths, use_container_width=True)
