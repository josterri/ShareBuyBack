# overview.py
import streamlit as st
import pandas as pd
from temp.benchmarks import vwap, twap, harmonic_mean

def run_overview_tab(df: pd.DataFrame):
    """
    Renders the Overview & Benchmark Metrics tab:
     - Price data preview (table + line chart)
     - VWAP, TWAP, Harmonic Mean metrics
    """
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
