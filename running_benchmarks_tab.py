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
      - Daily Volume (shares) & Daily Traded Value (USD)
    """
    # Prepare data
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2 = df2.sort_values('Date')

    # Compute running metrics
    df2['Running VWAP'] = (df2['Close'] * df2['Volume']).cumsum() / df2['Volume'].cumsum()
    df2['Running TWAP'] = df2['Close'].expanding().mean()
    df2['Daily Price'] = df2['Close']
    df2['Diff (bps)'] = (df2['Running VWAP'] - df2['Running TWAP']) \
                        / df2['Running TWAP'] * 10_000

    # 1) Price & running benchmarks
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(
        go.Scatter(x=df2['Date'], y=df2['Daily Price'], name='Daily Price', line=dict(color='black')),
        secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(x=df2['Date'], y=df2['Running VWAP'], name='Running VWAP', line=dict(color='blue')),
        secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(x=df2['Date'], y=df2['Running TWAP'], name='Running TWAP', line=dict(color='orange')),
        secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(x=df2['Date'], y=df2['Diff (bps)'], name='VWAP−TWAP (bps)',
                   line=dict(color='green', dash='dash')),
        secondary_y=True
    )
    fig1.update_layout(
        title='Price & Running VWAP/TWAP (with Diff in bps)',
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right')
    )
    fig1.update_yaxes(title_text='Price', secondary_y=False)
    fig1.update_yaxes(title_text='Difference (bps)', secondary_y=True)

    st.subheader("📊 Running Benchmarks")
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Daily Volume and Traded Value
    df2['Daily Value'] = df2['Close'] * df2['Volume']
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(
        go.Bar(x=df2['Date'], y=df2['Volume'], name='Daily Volume (shares)', marker_color='skyblue'),
        secondary_y=False
    )
    fig2.add_trace(
        go.Bar(x=df2['Date'], y=df2['Daily Value'], name='Daily Value (USD)', marker_color='lightgreen'),
        secondary_y=True
    )
    fig2.update_layout(
        title='Daily Traded Volume & Value',
        barmode='group',
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right')
    )
    fig2.update_yaxes(title_text='Volume (shares)', secondary_y=False)
    fig2.update_yaxes(title_text='Value (USD)', secondary_y=True)

    st.subheader("📊 Daily Volume & Traded Value")
    st.plotly_chart(fig2, use_container_width=True)
