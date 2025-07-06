# app.py 

import streamlit as st
from temp.monte_carlo_tab import run_monte_carlo_tab
from spend_averages_tab import run_spend_averages_tab
from temp.stop_after_tab     import run_stop_after_tab  # <-- new

st.set_page_config(page_title="Share Buyback Tool", layout="wide")
st.title("📈 Share Buyback Tool")

with st.sidebar:
    st.header("Monte Carlo Settings")
    mc_horiz      = st.number_input("Horizon (days)",        1, 2520, 125)
    mc_drift      = st.number_input("Drift (annual %)",      0.0, 100.0,   0.0, step=0.1)/100.0
    mc_vol        = st.number_input("Volatility (annual %)", 0.0, 100.0,  25.0, step=0.1)/100.0
    mc_sims       = st.number_input("Simulations",          1000, 100000, 10000, step=1000)
    initial_price = st.number_input("Initial Price (S₀)",     10, 1000,  100, step=10)
    total_shares  = st.number_input("Total Shares",            1, 1000000, 100, step=100)

tabs = st.tabs([
    "💲 Arithmetic vs Harmonic Mean"
])

with tabs[0]:
    run_spend_averages_tab(
        initial_price, mc_drift, mc_vol, mc_horiz, mc_sims, total_shares
    )
