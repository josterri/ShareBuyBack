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
st.set_page_config(page_title="Modular Buyback Tool", layout="wide")
st.title("Modular Share Buyback Pre-Trade Tool")

# ---- SIDEBAR: PARAMETERS ----
with st.sidebar:

    load_btn = st.button("Load / Generate Data")
    st.markdown("---")
    st.header("1. Monte Carlo")
    mc_sims   = st.number_input("Simulations", 1, 10_000, 500)
    mc_drift  = st.number_input("MC Drift (%)", 0.0, 100.0, 0.0) / 100.0
    mc_vol    = st.number_input("MC Vol (%)",   0.0, 100.0,15.0) / 100.0
    mc_horiz  = st.number_input("MC Days",       1, 2_520, 125)


    
    st.header("2. Data Source")
    source = st.radio("Load data from:", ["Synthetic GBM","Yahoo Finance", "Upload CSV"])
    
    # common params
    start = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end   = st.date_input("End Date",   value=pd.to_datetime("2023-12-31"))
    
    # source-specific
    if source == "Yahoo Finance":
        ticker = st.text_input("Ticker", "AAPL")
    elif source == "Upload CSV":
        upload = st.file_uploader("CSV file", type=["csv"])
    else:
        S0    = st.number_input("Initial Price", 10.0, 1e4, 100.0)
        mu    = st.number_input("Drift (%)", 0.0, 100.0, 0.0) / 100.0
        sigma = st.number_input("Volatility (%)", 0.0, 100.0, 15.0) / 100.0
        avgV  = st.number_input("Avg Volume", 1, 10_000_000, 1_000_000)

    st.markdown("---")
    st.header("3. Strategy Params")
    total_shares = st.number_input("TWAP: total shares", 1, 1_000_000, 10_000)
    vp_pct        = st.number_input("VP % of volume", 0.0, 100.0, 10.0) / 100.0


# ---- DATA LOADING FUNCTION ----
def load_data():
    if source == "Yahoo Finance":
        return fetch_yahoo(ticker, str(start), str(end))
    if source == "Upload CSV":
        return load_csv(upload)
    # Synthetic GBM
    return generate_gbm(start, end, S0, mu, sigma, avgV)

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
    tab_desc, tab_overview, tab_exec, tab_run, tab_mc  = st.tabs([
        "📊 Descriptive Analysis",
        "📈 Overview & Benchmarks",
        "⚙️ Execution Strategies",
        "📊 Running Benchmarks",
        "🎲 Monte Carlo Simulation"
    ])

    # --- TAB 1: Descriptive Analysis ---
    with tab_desc:
        run_descriptive_analysis(df)
    # --- TAB 2: Overview & Benchmarks ---
    with tab_overview:
        run_overview_tab(df)
    # --- TAB 3: Execution Strategies ---
    with tab_exec:
        run_execution_tab(df, total_shares, vp_pct)
        
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
            total_shares=total_shares,
            participation_frac=vp_pct
        )