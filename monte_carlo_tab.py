# monte_carlo_tab.py
import streamlit as st
from simulation import run_monte_carlo

def run_monte_carlo_tab(
    df,
    mc_drift,
    mc_vol,
    mc_horiz,
    mc_sims,
    total_shares,
    participation_frac
):
    """
    Renders the Monte Carlo Simulation tab:
      - Runs run_monte_carlo with provided parameters
      - Displays summary stats, bar chart, and download button
    """
    st.subheader("Monte Carlo Cost Distribution")

    # Last observed price & avg volume
    S0_last = df['Close'].iloc[-1]
    avg_vol = df['Volume'].mean()

    # Run simulations
    costs_df = run_monte_carlo(
        S0=S0_last,
        avg_vol=avg_vol,
        mu=mc_drift,
        sigma=mc_vol,
        horizon=mc_horiz,
        sims=mc_sims,
        total_shares=total_shares,
        participation_frac=participation_frac
    )

    # Show descriptive stats and chart
    st.write(costs_df.describe())
    st.bar_chart(costs_df, height=300)

    # Download CSV
    csv = costs_df.to_csv(index=False)
    st.download_button(
        label="Download MC Results",
        data=csv,
        file_name="mc_costs.csv",
        mime="text/csv"
    )
