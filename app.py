# app.py
import streamlit as st
import pandas as pd

from data_loader import fetch_yahoo, load_csv, generate_gbm
from benchmarks import vwap, twap, harmonic_mean
from strategies import simulate_twap, simulate_volume_participation
from simulation import run_monte_carlo
from descriptive import run_descriptive_analysis
from strategies_tab import run_execution_tab
from monte_carlo_tab import run_monte_carlo_tab
from running_benchmarks_tab import run_running_benchmarks_tab
from overview import run_overview_tab

from simulation import run_monte_carlo
from data_loader import fetch_yahoo, load_csv, generate_gbm
# ---- UI CONFIG ----
st.set_page_config(page_title="Share Buyback Tool", layout="wide")
st.title("Share Buyback Pre-Trade Tool")

# ---- SIDEBAR: PARAMETERS ----
with st.sidebar:
    
    st.markdown("---")
    load_btn = st.button("Start Computation")
    st.markdown("---")

    st.header("Monte Carlo Simulation Parameters")
    mc_sims = st.number_input("Simulations", min_value=1, max_value=10_000, value=100)
    mc_drift = st.number_input("Drift (annualized %)", min_value=0.0, max_value=100.0, value=0.0, step=0.1) / 100.0
    mc_vol = st.number_input("Volatility (annualized %)", min_value=0.0, max_value=100.0, value=25.0, step=0.1) / 100.0
    mc_horiz = st.number_input("Horizon (days)", min_value=1, max_value=2520, value=125)

    st.markdown("---")
    st.header("Synthetic Data (GBM)")

    start = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    S0 = st.number_input("Initial Price", min_value=1.0, max_value=10_000.0, value=100.0)
  #  mu = st.number_input("Drift (GBM %) ", min_value=-50.0, max_value=100.0, value=0.0, step=0.1) / 100.0
#    sigma = st.number_input("Volatility (GBM %)", min_value=0.0, max_value=100.0, value=15.0, step=0.1) / 100.0
    avgV = st.number_input("Average Daily Volume", min_value=1, max_value=10_000_000, value=1_000_000)

    st.markdown("---")
    st.header("Execution Strategy")
    total_shares = st.number_input("Total Shares (TWAP)", min_value=1, max_value=1_000_000, value=10_000)
 #   vp_pct = st.number_input("Participation Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.5) / 100.0

    st.markdown("https://www.joergosterrieder.com/")

# ---- DATA LOADING FUNCTION ----
def load_data():
    return generate_gbm(start, end, S0, mc_drift, mc_vol, avgV)

# ---- MAIN ----
if load_btn:
    try:
        df = load_data()
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Data load error: {e}")

if 'df' in st.session_state:
    df = st.session_state['df']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # ---- TABS ----
#    tab_mc, tab_desc, tab_overview, tab_exec, tab_run= st.tabs([
    tab_mc, tab_desc, tab_run= st.tabs([
        "🎲 TWAP vs Fixed-Notional",
        "📊 Example - Descriptive Analysis",
#        "📈 Overview & Benchmarks",
   #     "⚙️ Execution Strategies",
        "📊 Example - Running Benchmarks"
    ])

    # --- TAB 1: Descriptive Analysis ---
    with tab_desc:
        run_descriptive_analysis(df)
    # --- TAB 2: Overview & Benchmarks ---
#   with tab_overview:
 #       run_overview_tab(df)
    # --- TAB 3: Execution Strategies ---
  #  with tab_exec:
   #     run_execution_tab(df, total_shares, vp_pct)
        
    # --- TAB 4: Running Benchmarks ---
    with tab_run:
        run_running_benchmarks_tab(df)
    # --- TAB 5: Monte Carlo Simulation ---
    with tab_mc:
        run_monte_carlo_tab(
            df,
            mc_drift=mc_drift,
            mc_vol=mc_vol,
            mc_horiz=mc_horiz,
            mc_sims=mc_sims,
            total_shares=total_shares
        )