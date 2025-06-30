# running_benchmarks_tab.py
import streamlit as st
import pandas as pd

def run_running_benchmarks_tab(df: pd.DataFrame):
    """
    Plots:
      - Daily Price
      - Cumulative (running) VWAP
      - Cumulative (running) TWAP (simple average)
    """
    # Prepare
    df2 = df.copy()
    df2 = df2.sort_values('Date')
    # Running VWAP
    cum_pv = (df2['Close'] * df2['Volume']).cumsum()
    cum_vol = df2['Volume'].cumsum()
    df2['Running VWAP'] = cum_pv / cum_vol
    # Running TWAP = expanding mean of price
    df2['Running TWAP'] = df2['Close'].expanding().mean()
    # Daily Price
    df2['Daily Price'] = df2['Close']
    
    # Select and reindex
    plot_df = df2.set_index('Date')[['Daily Price','Running VWAP','Running TWAP']]
    
    st.subheader("📊 Running Benchmarks")
    st.line_chart(plot_df, use_container_width=True)
