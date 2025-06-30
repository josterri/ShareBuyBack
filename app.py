# app.py
import streamlit as st
import pandas as pd

from data_loader import fetch_yahoo, load_csv, generate_gbm
from benchmarks import vwap, twap, harmonic_mean
from strategies import simulate_twap, simulate_volume_participation
from simulation import run_monte_carlo
from descriptive import run_descriptive_analysis

# ---- UI CONFIG ----
st.set_page_config(page_title="Modular Buyback Tool", layout="wide")
st.title("Modular Share Buyback Pre-Trade Tool")

# ---- SIDEBAR: PARAMETERS ----
with st.sidebar:
    st.header("1. Data Source")
    source = st.radio("Load data from:", ["Yahoo Finance", "Upload CSV", "Synthetic GBM"])
    
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
        mu    = st.number_input("Drift (%)", 0.0, 100.0, 5.0) / 100.0
        sigma = st.number_input("Volatility (%)", 0.0, 100.0, 20.0) / 100.0
        avgV  = st.number_input("Avg Volume", 1, 10_000_000, 1_000_000)

    st.markdown("---")
    st.header("2. Strategy Params")
    total_shares = st.number_input("TWAP: total shares", 1, 1_000_000, 10_000)
    vp_pct        = st.number_input("VP % of volume", 0.0, 100.0, 10.0) / 100.0

    st.markdown("---")
    st.header("3. Monte Carlo")
    mc_sims   = st.number_input("Simulations", 1, 10_000, 500)
    mc_drift  = st.number_input("MC Drift (%)", 0.0, 100.0, 5.0) / 100.0
    mc_vol    = st.number_input("MC Vol (%)",   0.0, 100.0,20.0) / 100.0
    mc_horiz  = st.number_input("MC Days",       1, 2_520, 252)

    load_btn = st.button("Load / Generate Data")

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
    tab_desc, tab_overview, tab_exec, tab_mc = st.tabs([
        "📊 Descriptive Analysis",
        "📈 Overview & Benchmarks",
        "⚙️ Execution Strategies",
        "🎲 Monte Carlo Simulation"
    ])

    # --- TAB 1: Descriptive Analysis ---
    with tab_desc:
        run_descriptive_analysis(df)
    # --- TAB 2: Overview & Benchmarks ---
    with tab_overview:
        st.subheader("Price Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.line_chart(df.set_index('Date')['Close'], height=250)

        st.subheader("Benchmark Metrics")
        vw = vwap(df['Close'], df['Volume'])
        tw = twap(df['Close'])
        hm = harmonic_mean(df['Close'])
        c1, c2, c3 = st.columns(3)
        c1.metric("VWAP", f"{vw:.3f}")
        c2.metric("TWAP (mean)", f"{tw:.3f}")
        c3.metric("Harmonic Mean", f"{hm:.3f}")

    # --- TAB 3: Execution Strategies ---
    with tab_exec:
        st.subheader("TWAP Strategy")
        twap_df = simulate_twap(df, total_shares)
        st.dataframe(twap_df, use_container_width=True)
        st.download_button("Download TWAP CSV",
                           twap_df.to_csv(index=False),
                           "twap_log.csv")

        st.subheader("Volume Participation Strategy")
        vp_df = simulate_volume_participation(df, vp_pct)
        st.dataframe(vp_df, use_container_width=True)
        st.download_button("Download VP CSV",
                           vp_df.to_csv(index=False),
                           "vp_log.csv")

    # --- TAB 4: Monte Carlo Simulation ---
    with tab_mc:
        st.subheader("Monte Carlo Cost Distribution")
        S0_last = df['Close'].iloc[-1]
        avg_vol = df['Volume'].mean()
        costs_df = run_monte_carlo(
            S0=S0_last,
            avg_vol=avg_vol,
            mu=mc_drift,
            sigma=mc_vol,
            horizon=mc_horiz,
            sims=mc_sims,
            total_shares=total_shares,
            participation_frac=vp_pct
        )
        st.write(costs_df.describe())
        st.bar_chart(costs_df, height=300)
        st.download_button("Download MC Results",
                           costs_df.to_csv(index=False),
                           "mc_costs.csv")
