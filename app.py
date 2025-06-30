# app.py
import streamlit as st
import pandas as pd

from data_loader import fetch_yahoo, load_csv, generate_gbm
from benchmarks import vwap, twap, harmonic_mean
from strategies import simulate_twap, simulate_volume_participation
from simulation import run_monte_carlo

st.set_page_config(page_title="Modular Buyback Tool", layout="wide")
st.title("Modular Share Buyback Pre-Trade Tool")

# --- Sidebar: data source & parameters ---
src = st.sidebar.radio("Data Source", ["Yahoo Finance","Upload CSV","Synthetic GBM"])

if src=="Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    start = st.sidebar.date_input("Start", pd.to_datetime("2023-01-01"))
    end   = st.sidebar.date_input("End",   pd.to_datetime("2023-12-31"))
elif src=="Upload CSV":
    upload = st.sidebar.file_uploader("CSV file", type=["csv"])
else:
    start = st.sidebar.date_input("Start", pd.to_datetime("2023-01-01"))
    end   = st.sidebar.date_input("End",   pd.to_datetime("2023-12-31"))
    S0    = st.sidebar.number_input("Init Price", 1.0, 1000.0, 100.0)
    mu    = st.sidebar.number_input("Drift (%)", 0.0, 100.0, 5.0)/100.0
    sigma = st.sidebar.number_input("Vol (%)",  0.0, 100.0,20.0)/100.0
    avgV  = st.sidebar.number_input("Avg Volume", 1, 10000000, 1_000_000)

# Strategy params
st.sidebar.header("Strategy Params")
total_shares = st.sidebar.number_input("TWAP: total shares", 1, 1000000, 10000)
vp_pct        = st.sidebar.number_input("VP % of volume", 0.0, 100.0, 10.0)/100.0

# Monte Carlo params
st.sidebar.header("Monte Carlo")
mc_sims   = st.sidebar.number_input("Simulations",1,10000,500)
mc_drift  = st.sidebar.number_input("MC drift %",0.0,100.0,5.0)/100.0
mc_vol    = st.sidebar.number_input("MC vol %", 0.0,100.0,20.0)/100.0
mc_horiz  = st.sidebar.number_input("MC days",1,2520,252)

if st.sidebar.button("Load / Generate Data"):
    try:
        if src=="Yahoo Finance":
            df = fetch_yahoo(ticker, str(start), str(end))
        elif src=="Upload CSV":
            df = load_csv(upload)
        else:
            df = generate_gbm(start, end, S0, mu, sigma, avgV)
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Data error: {e}")

# Once data is loaded
if 'df' in st.session_state:
    df = st.session_state['df']
    # show head
    st.subheader("Loaded Price Data")
    st.dataframe(df.head())

    # benchmarks
    st.subheader("Benchmarks")
    st.write("VWAP:", vwap(df['Close'], df['Volume']))
    st.write("TWAP:", twap(df['Close']))
    st.write("Harmonic mean:", harmonic_mean(df['Close']))

    # single-run strategies
    tab1, tab2 = st.tabs(["TWAP","Volume Participation"])
    with tab1:
        tw = simulate_twap(df, total_shares)
        st.dataframe(tw)
        st.download_button("Download TWAP CSV", tw.to_csv(index=False), "twap.csv")
    with tab2:
        vp = simulate_volume_participation(df, vp_pct)
        st.dataframe(vp)
        st.download_button("Download VP CSV", vp.to_csv(index=False), "vp.csv")

    # Monte Carlo
    st.subheader("Monte Carlo Cost Distribution")
    costs = run_monte_carlo(
        df['Close'].iloc[-1],
        df['Volume'].mean(),
        mc_drift, mc_vol,
        mc_horiz, mc_sims,
        total_shares, vp_pct
    )
    st.write(costs.describe())
    st.bar_chart(costs)

    st.download_button("Download MC results", costs.to_csv(index=False), "mc_costs.csv")
