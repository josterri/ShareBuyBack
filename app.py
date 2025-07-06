import streamlit as st
import pandas as pd

from data_loader import generate_gbm
from descriptive import run_descriptive_analysis
from running_benchmarks_tab import run_running_benchmarks_tab
from monte_carlo_tab import run_monte_carlo_tab

# ---- UI CONFIG ----
st.set_page_config(page_title="Share Buyback Pre-Trade Tool", layout="wide")
st.title("📈 Share Buyback Pre-Trade Tool")

st.markdown(
    """
    **Welcome!**  
    Configure your synthetic data, Monte Carlo simulation, and execution strategy in the sidebar,  
    then click **Start Simulation** to generate results.
    """
)

# ---- SIDEBAR: PARAMETERS ----
with st.sidebar:
    st.header("1. Synthetic Data (GBM)")
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date   = st.date_input("End Date",   pd.to_datetime("2023-12-31"))
    init_price = st.number_input("Initial Price (S₀)", 1.0, 10_000.0, 100.0, step=1.0)
    drift_gbm  = st.number_input("GBM Drift (%)", 0.0, 100.0, 0.0, step=0.1) / 100.0
    vol_gbm    = st.number_input("GBM Volatility (%)", 0.0, 100.0, 15.0, step=0.1) / 100.0
    avg_volume = st.number_input("Avg Daily Volume", 1, 10_000_000, 1_000_000, step=1000)
    st.markdown("---")
    st.header("2. Monte Carlo Simulation")
    mc_sims  = st.number_input("Simulations",         1, 10_000, 500)
    mc_horiz = st.number_input("Horizon (days)",      1, 2_520, 125)
    mc_drift = st.number_input("Drift (annual %) ",  0.0, 100.0,   0.0, step=0.1) / 100.0
    mc_vol   = st.number_input("Volatility (annual %)", 0.0, 100.0, 25.0, step=0.1) / 100.0

    st.markdown("---")
    st.header("3. Execution Strategy")
    total_shares = st.number_input("Total Shares (TWAP)", 1, 1_000_000, 10_000, step=100)

    st.markdown("---")
    run_btn = st.button("Start Simulation")

# ---- DATA LOADING FUNCTION ----
def load_data():
    return generate_gbm(
        start_date, end_date,
        init_price, drift_gbm,
        vol_gbm, avg_volume
    )

# ---- MAIN ----
if run_btn:
    try:
        df = load_data()
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Error generating data: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    tab_mc, tab_desc, tab_run = st.tabs([
        "🎲 Monte Carlo",
        "📊 Descriptive Analysis",
        "📊 Running Benchmarks"
    ])

    with tab_mc:
        st.markdown("## Monte Carlo Simulation")
        st.markdown(
            "Simulate GBM price paths and compare two execution strategies:\n\n"
            "- **TWAP**: fixed shares per day\n"
            "- **Fixed-Notional**: constant USD per day"
        )
        run_monte_carlo_tab(
            df=df,
            initial_price=init_price,
            avg_vol=avg_volume,
            mc_drift=mc_drift,
            mc_vol=mc_vol,
            mc_horiz=mc_horiz,
            mc_sims=mc_sims,
            total_shares=total_shares,
        )

    with tab_desc:
        st.markdown("## Descriptive Analysis")
        run_descriptive_analysis(df)

    with tab_run:
        st.markdown("## Running Benchmarks")
        run_running_benchmarks_tab(df)

st.sidebar.markdown("Built by [Joerg Osterrieder](https://www.joergosterrieder.com)")
