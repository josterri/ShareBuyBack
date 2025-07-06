# app.py
import streamlit as st
import pandas as pd

from data_loader import generate_gbm  # or however you get your price series
from monte_carlo_tab import run_monte_carlo_tab

st.set_page_config(page_title="Share Buyback Tool", layout="wide")
st.title("📈 Share Buyback Pre-Trade Tool")

# ---- SIDEBAR: Monte Carlo only ----
with st.sidebar:
    st.header("Monte Carlo Simulation")
    mc_horiz = st.number_input("Horizon (days)",        1, 2520, 125)
    mc_drift = st.number_input("Drift (annual %)",      0.0, 100.0,  0.0, step=0.1) / 100.0
    mc_vol   = st.number_input("Volatility (annual %)", 0.0, 100.0, 25.0, step=0.1) / 100.0

    st.markdown("---")
    run_btn = st.button("▶️ Start Simulation")

# ---- MAIN ----
if run_btn:
    # if you need to generate synthetic data:
    # df = generate_gbm(...)
    # otherwise, load your actual df from elsewhere
    df = generate_gbm(  # for example
        pd.to_datetime("2023-01-01"),
        pd.to_datetime("2023-12-31"),
        S0=100.0,
        mu=0.0,
        sigma=0.15,
    )
    st.session_state["df"] = df

if "df" in st.session_state:
    df = st.session_state["df"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    tab_mc, tab_dummy = st.tabs([
        "🎲 Monte Carlo",
        "📊 (other tabs…)"])
    
    with tab_mc:
        run_monte_carlo_tab(
            df=df,
            initial_price=df["Close"].iloc[-1],  # or pass your own
            mc_drift=mc_drift,
            mc_vol=mc_vol,
            mc_horiz=mc_horiz,
            mc_sims=500,            # you can hard-code or expose sims if you want
            total_shares=10_000     # same for total_shares
        )

    with tab_dummy:
        st.write("…")
