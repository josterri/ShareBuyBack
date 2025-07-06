# monte_carlo_tab.py

import streamlit as st
import pandas as pd
import plotly.express as px
from mcsimulation import load_or_run_simulation
import numpy as np
def run_monte_carlo_tab(
    initial_price: float,
    mc_drift: float,
    mc_vol: float,
    mc_horiz: int,
    mc_sims: int,
    total_shares: float = 100
):
    """
    Tab 1: Compare TWAP vs Fixed-Notional in bps, and show GBM paths.
    """
    st.markdown(
        f"**S₀:** {initial_price:.0f}  |  "
        f"**Drift:** {mc_drift:.1%}  |  "
        f"**Vol:** {mc_vol:.1%}  |  "
        f"**Horizon:** {mc_horiz}d  |  "
        f"**Sims:** {mc_sims:,}"
    )

    # load or run & cache
    df_px = load_or_run_simulation(
        initial_price, mc_drift, mc_vol, mc_horiz, mc_sims, total_shares
    )

    # 1) Price‐outperformance in basis‐points
    bps_diff = (df_px["P_twap"] - df_px["P_usd"]) / df_px["P_twap"] * 1e4
    st.subheader("1. Price Outperformance (bps)")
    st.metric("Mean Δ (bps)", f"{bps_diff.mean():.1f}", f"σ={bps_diff.std():.1f}")

    df_hist = pd.DataFrame({"Δ in bps": bps_diff})
    fig = px.histogram(df_hist, x="Δ in bps", nbins=200)
    fig.update_layout(xaxis_title="bps Δ", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # 2) Simulated GBM price paths (only 100 for plotting)
    st.subheader("2. Simulated GBM Price Paths")
    n_plot = min(mc_sims, 100)
    mu_d, sigma_d = mc_drift/252.0, mc_vol/np.sqrt(252.0)
    Z_small = np.random.normal(size=(mc_horiz, n_plot))
    logs_small = np.vstack([
        np.zeros(n_plot),
        np.cumsum((mu_d - 0.5*sigma_d**2) + sigma_d * Z_small, axis=0)
    ])
    price_paths_small = initial_price * np.exp(logs_small)

    df_paths = (
        pd.DataFrame(price_paths_small)
          .reset_index()
          .melt(id_vars="index", var_name="Simulation", value_name="Price")
          .rename(columns={"index": "Day"})
    )
    fig2 = px.line(
        df_paths, x="Day", y="Price",
        color="Simulation", line_group="Simulation"
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)