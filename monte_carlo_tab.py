# monte_carlo_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from simulation import run_monte_carlo

@st.cache_data(show_spinner=False)
def cached_run_mc(S0, avg_vol, mu, sigma, horizon, sims, total_shares, participation_frac):
    return run_monte_carlo(
        S0=S0,
        avg_vol=avg_vol,
        mu=mu,
        sigma=sigma,
        horizon=horizon,
        sims=sims,
        total_shares=total_shares,
        participation_frac=participation_frac
    )

def run_monte_carlo_tab(
    df: pd.DataFrame,
    mc_drift: float,
    mc_vol: float,
    mc_horiz: int,
    mc_sims: int,
    total_shares: float,
    participation_frac: float
):
    """
    Monte Carlo Simulation Analysis:
      1. Price-based outperformance (bps)
      2. Shares difference (USD-daily vs TWAP)
      3. Execution cost distribution
      4. Simulated price paths
    """

    # Prepare constants
    S0      = df['Close'].iloc[-1]
    avg_vol = df['Volume'].mean()

    st.markdown(
        f"## Monte Carlo Setup\n"
        f"- **Initial Price (S₀):** ${S0:.2f}\n"
        f"- **Simulations:** {mc_sims}\n"
        f"- **Horizon:** {mc_horiz} days\n"
        f"- **Drift:** {mc_drift:.1%} annual\n"
        f"- **Volatility:** {mc_vol:.1%} annual\n"
    )

    # Run MC once
    costs_df = cached_run_mc(
        S0, avg_vol,
        mc_drift, mc_vol,
        mc_horiz, mc_sims,
        total_shares, participation_frac
    )

    # Identify cost columns dynamically
    cost_cols = list(costs_df.columns)
    twap_col, vp_col = cost_cols

    # Rebuild price paths for the two outperformance measures
    mu_d = mc_drift / 252.0
    sigma_d = mc_vol / np.sqrt(252.0)
    Z = np.random.normal(size=(mc_horiz, mc_sims))
    increments = (mu_d - 0.5 * sigma_d**2) + sigma_d * Z
    cum_logs = np.vstack([np.zeros(mc_sims), np.cumsum(increments, axis=0)])
    price_paths = S0 * np.exp(cum_logs)

    # 1) Price Advantage in Basis Points
    st.subheader("1. Price Advantage in Basis Points")
    
    # Explanation text
    st.markdown("""
    Quantify the difference in average execution price between:
    
    - **TWAP** (fixed‐shares per day)  
    - **Fixed‐Notional** (constant USD per day)
    
    and express it in basis points (1 bp = 0.01 %).
    """)
    
    # Define P_TWAP and P_USD
    st.markdown("**Define average execution prices:**")
    st.latex(r"""
    P_{\mathrm{TWAP}}
    = \frac{\text{Total cost under TWAP}}{\text{Total shares under TWAP}}
    \quad,\quad
    P_{\mathrm{USD}}
    = \frac{\text{Total cost under Fixed-Notional}}{\text{Total shares under Fixed-Notional}}
    """)
    
    # Define delta_bps
    st.markdown("**Compute basis‐point difference:**")
    st.latex(r"""
    \Delta_{\mathrm{bps}}
    = \frac{P_{\mathrm{TWAP}} - P_{\mathrm{USD}}}{P_{\mathrm{TWAP}}}
    \times 10\,000
    """)
    
    st.markdown("""
    - $\Delta_{\\mathrm{bps}} > 0$: USD-daily paid a lower average price (outperformed TWAP).
    - $\Delta_{\\mathrm{bps}} < 0$: USD-daily paid a higher average price (underperformed TWAP).
    """)
    

    # calculate USD vs TWAP
    daily_notional = total_shares * S0 / mc_horiz
    shares_usd     = (daily_notional / price_paths[1:]).sum(axis=0)
    fixed_avg_price = price_paths[1:].mean(axis=0)
    usd_avg_price   = (total_shares * S0) / shares_usd
    bps_diff        = (fixed_avg_price - usd_avg_price) / fixed_avg_price * 1e4

    # 2) Price-based Outperformance (bps)
    st.subheader("2. Price Difference in Basis Points")
    
    # build a DataFrame so Plotly picks up the column name
    df_bps = pd.DataFrame({
        'Price Outperformance (bps)': bps_diff
    })
    
    fig_bps = px.histogram(
        df_bps,
        x='Price Outperformance (bps)',
        nbins=200,
        title='Distribution of Price Outperformance (bps)'
    )
    fig_bps.update_layout(
        xaxis_title='Price Outperformance (bps)',
        yaxis_title='Frequency'
    )
    
    st.plotly_chart(fig_bps, use_container_width=True)
    
    b_mean, b_std = bps_diff.mean(), bps_diff.std()
    st.metric("Mean bps Δ", f"{b_mean:.1f}", f"σ={b_std:.1f}")


    # 4) Simulated Price Paths
    st.subheader("4. Simulated Price Paths")
    st.markdown("All simulated GBM price paths over the horizon.")
    df_paths = pd.DataFrame(price_paths).reset_index().melt(
        id_vars='index', var_name='simulation', value_name='price'
    ).rename(columns={'index': 'Day'})
    fig_paths = px.line(
        df_paths,
        x='Day', y='price', color='simulation',
        line_group='simulation', opacity=0.15,
        labels={'price': 'Price', 'Day': 'Day'},
        title="Simulated GBM Price Paths"
    )
    fig_paths.update_layout(showlegend=False)
    st.plotly_chart(fig_paths, use_container_width=True)
