# monte_carlo_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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


def run_historical_execution_chart(df: pd.DataFrame, total_shares: float):
    """
    Plots:
      - Daily Price
      - Running Average Execution Price under:
          • Fixed-Shares (TWAP)
          • Fixed-Notional (constant USD)
      - Secondary axis: Price difference in bps between the two
    """
    # 1) Compute running TWAP (fixed shares)
    n = len(df)
    fs_per_day = total_shares / n
    cum_shares_fs = np.arange(1, n+1) * fs_per_day
    cum_cost_fs   = (df['Close'] * fs_per_day).cumsum()
    avg_price_fs  = cum_cost_fs / cum_shares_fs

    # 2) Compute running USD strategy
    S0 = df['Close'].iloc[0]
    usd_notional = total_shares * S0 / n
    shares_usd    = usd_notional / df['Close']
    cum_shares_usd = shares_usd.cumsum()
    cum_cost_usd   = usd_notional * np.arange(1, n+1)
    avg_price_usd  = cum_cost_usd / cum_shares_usd

    # 3) bps difference
    bps_diff = (avg_price_fs - avg_price_usd) / avg_price_fs * 1e4

    # assemble into one DataFrame
    df_exec = pd.DataFrame({
        'Date':        df['Date'],
        'Daily Price': df['Close'],
        'Fixed-Shares Price': avg_price_fs,
        'Fixed-Notional Price': avg_price_usd,
        'Diff (bps)':  bps_diff
    }).set_index('Date')

    # 4) Plot with secondary axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df_exec.index, y=df_exec['Daily Price'],
                   name='Daily Price', line=dict(color='black')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_exec.index, y=df_exec['Fixed-Shares Price'],
                   name='Fixed-Shares', line=dict(color='orange')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_exec.index, y=df_exec['Fixed-Notional Price'],
                   name='Fixed-Notional', line=dict(color='blue')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df_exec.index, y=df_exec['Diff (bps)'],
                   name='Price Δ (bps)', line=dict(color='green', dash='dash')),
        secondary_y=True
    )

    fig.update_layout(
        title="Price & Running Execution Prices (with Δ in bps)",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Difference (bps)", secondary_y=True)

    st.subheader("One Execution Example")
    st.markdown(
        "This chart shows one price series along with the running average execution price\n"
        "if you had bought **fixed shares** each day versus **fixed USD** each day, and the bps difference."
    )
    st.plotly_chart(fig, use_container_width=True)

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
 # ——— Introduction ———
    st.markdown("# TWAP vs Fixed-Notional")
    st.markdown(
        "This section runs a Geometric Brownian Motion simulation for future prices,\n"
        "then compares two execution strategies:\n\n"
        "1. **TWAP** (fixed‐shares per day)\n"
        "2. **Fixed-Notional** (constant USD per day)\n\n"
        "We report:\n"
        "- Price‐based outperformance in basis points\n"
        "- All simulated price paths"
    )
    
    
    # Prepare constants
    S0      = df['Close'].iloc[-1]
    avg_vol = df['Volume'].mean()
    
     # Run MC once
    costs_df = cached_run_mc(
        S0, avg_vol,
        mc_drift, mc_vol,
        mc_horiz, mc_sims,
        total_shares, participation_frac
    )

    st.markdown(
        f"- **Horizon:** {mc_horiz} days\n"
        f"- **Drift:** {mc_drift:.1%} annual\n"
        f"- **Volatility:** {mc_vol:.1%} annual\n"
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
  # calculate USD vs TWAP
    daily_notional = total_shares * S0 / mc_horiz
    shares_usd     = (daily_notional / price_paths[1:]).sum(axis=0)
    fixed_avg_price = price_paths[1:].mean(axis=0)
    usd_avg_price   = (total_shares * S0) / shares_usd
    bps_diff        = (fixed_avg_price - usd_avg_price) / fixed_avg_price * 1e4

    # 2) Price-based Outperformance (bps)
    st.subheader("1. Price Difference in Basis Points")
    
    # build a DataFrame so Plotly picks up the column name
    df_bps = pd.DataFrame({
        'Price Outperformance (bps)': bps_diff
    })
    
    fig_bps = px.histogram(
        df_bps,
        x='Price Outperformance (bps)',
        nbins=200,
        title='Distribution of Price Outperformance (bps) vs TWAP'
    )
    fig_bps.update_layout(
        xaxis_title='Price Outperformance (bps)',
        yaxis_title='Frequency'
    )
    
    st.plotly_chart(fig_bps, use_container_width=True)
    
    b_mean, b_std = bps_diff.mean(), bps_diff.std()

    # After computing b_mean, b_std, s_mean, s_std…

    # ——— Key Findings ———
    st.markdown("### Key Findings")
    st.markdown(f"""
    - **Average price advantage:** USD-daily outperforms TWAP by **{b_mean:.1f} bps** on average  
      (σ = {b_std:.1f} bps).
    """)
    st.metric("Mean bps Δ", f"{b_mean:.1f}", f"σ={b_std:.1f}")

    run_historical_execution_chart(df, total_shares)
   
    # 1) Price Advantage in Basis Points
    st.subheader("2. Price Advantage in Basis Points")
    
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
    

    # 3) Simulated Price Paths
    st.subheader("3. Simulated Price Paths")
    st.markdown("""
Plot up to 100 simulated price trajectories under the GBM model to illustrate
the range and variability of possible future paths over the specified horizon.
""")

    # Limit to at most 100 paths for plotting
    max_paths = min(price_paths.shape[1], 100)
    paths_to_plot = price_paths[:, :max_paths]

    df_paths = (
        pd.DataFrame(paths_to_plot)
          .reset_index()
          .melt(id_vars='index', var_name='simulation', value_name='price')
          .rename(columns={'index': 'Day'})
    )

    fig_paths = px.line(
        df_paths,
        x='Day', y='price', color='simulation',
        line_group='simulation', 
        labels={'price': 'Price', 'Day': 'Day'},
        title=f"Simulated GBM Price Paths (showing {max_paths} of {price_paths.shape[1]} sims)"
    )
    fig_paths.update_layout(showlegend=False)
    st.plotly_chart(fig_paths, use_container_width=True)
