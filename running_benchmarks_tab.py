# running_benchmarks_tab.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_running_benchmarks_tab(df: pd.DataFrame):
    """
    Plots using Plotly:
      - Daily Price
      - Cumulative (running) VWAP
      - Cumulative (running) TWAP (simple average)
      - Secondary axis: Difference between VWAP & TWAP in basis points
    """
    # Prepare data
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2 = df2.sort_values('Date')

    # Compute running metrics
    df2['Running VWAP'] = (df2['Close'] * df2['Volume']).cumsum() / df2['Volume'].cumsum()
    df2['Running TWAP'] = df2['Close'].expanding().mean()
    df2['Daily Price'] = df2['Close']

    # Compute difference in basis points
    # bps = (VWAP - TWAP) / TWAP * 10,000
    df2['Diff (bps)'] = (df2['Running VWAP'] - df2['Running TWAP']) \
                        / df2['Running TWAP'] * 10_000

    # Build figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary series: prices and benchmarks
    fig.add_trace(
        go.Scatter(
            x=df2['Date'], y=df2['Daily Price'],
            name='Daily Price', line=dict(color='black')
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df2['Date'], y=df2['Running VWAP'],
            name='Running VWAP', line=dict(color='blue')
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df2['Date'], y=df2['Running TWAP'],
            name='Running TWAP', line=dict(color='orange')
        ),
        secondary_y=False
    )

    # Secondary series: difference in bps
    fig.add_trace(
        go.Scatter(
            x=df2['Date'], y=df2['Diff (bps)'],
            name='VWAP − TWAP (bps)', line=dict(color='green', dash='dash')
        ),
        secondary_y=True
    )

    # Layout
    fig.update_layout(
        title='Daily Price with Running VWAP, TWAP, and VWAP−TWAP (bps)',
        xaxis_title='Date',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_yaxes(title_text='Price', secondary_y=False)
    fig.update_yaxes(title_text='Difference (bps)', secondary_y=True)

    # Render
    st.subheader("📊 Running Benchmarks")
    st.plotly_chart(fig, use_container_width=True)
