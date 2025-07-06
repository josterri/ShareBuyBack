import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from mcsimulation import load_or_run_simulation

def run_stop_after_tab(
    initial_price: float,
    mc_drift: float,
    mc_vol: float,
    mc_horiz: int,
    mc_sims: int,
    total_shares: float = 100
):
    """
    Tab 3: Stop‐After Strategy
      - Spend a fixed $ each day (we use daily_notional = TotalShares·S₀/H here)
      - Stop buying once cumulative shares ≥ H·S₀
    """
    st.markdown("## 3. Stop‐After Strategy")
    threshold_shares = mc_horiz * initial_price
    st.markdown(f"We stop once we've bought **{threshold_shares:.0f}** shares (H×S₀).")

    # 1) simulate GBM again (to get fresh paths for this tab)
    mu_d, sigma_d = mc_drift/252.0, mc_vol/np.sqrt(252.0)
    Z    = np.random.normal(size=(mc_horiz, mc_sims))
    logs = np.vstack([
        np.zeros(mc_sims),
        np.cumsum((mu_d - 0.5*sigma_d**2) + sigma_d * Z, axis=0)
    ])
    price_paths = initial_price * np.exp(logs)  # shape (H+1, sims)

    # 2) daily notional spending (same as fixed‐notional)
    daily_notional = total_shares * initial_price / mc_horiz

    # 3) for each sim, buy until threshold_shares is reached
    avg_price_stop = np.zeros(mc_sims)
    for i in range(mc_sims):
        cum_shares = 0.0
        cum_cost   = 0.0
        for t in range(mc_horiz):
            if cum_shares >= threshold_shares:
                break
            price = price_paths[t+1, i]
            spend = daily_notional
            shares = spend / price
            cum_shares += shares
            cum_cost   += spend
        # actual avg price paid per share
        avg_price_stop[i] = cum_cost / cum_shares

    # 4) get TWAP avg prices (cached)
    df_px = load_or_run_simulation(
        initial_price, mc_drift, mc_vol, mc_horiz, mc_sims, total_shares
    )
    P_twap = df_px["P_twap"].values

    # 5) bps difference vs TWAP
    bps_stop = (P_twap - avg_price_stop) / P_twap * 1e4

    st.subheader("Stop‐After vs TWAP (bps)")
    st.metric("Mean Δ (bps)", f"{bps_stop.mean():.1f}", f"σ={bps_stop.std():.1f}")

    df_hist = pd.DataFrame({"Δ_stop (bps)": bps_stop})
    fig = px.histogram(df_hist, x="Δ_stop (bps)", nbins=200)
    fig.update_layout(xaxis_title="bps Δ", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
