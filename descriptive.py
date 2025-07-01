# descriptive.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_descriptive_analysis(df: pd.DataFrame):
    """
    Performs and renders a full descriptive analysis of df['Close'] using Plotly:
     1. Price time series
     2. Histogram of daily returns
     3. Rolling 20-day moving average & volatility
     4. Summary statistics table (no decimals, drop NaNs)
     5. Price distribution histogram
    """
    # Prepare data
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2.sort_values('Date', inplace=True)
    prices = df2['Close']
    returns = prices.pct_change().dropna()
    volumes = df2.get('Volume', pd.Series(dtype=float))

    # Sections 1–3 unchanged: time series, return histogram, rolling stats
    # …

    # 4) Summary statistics
    st.subheader("Summary Statistics")
    raw_desc = prices.describe()  # contains 'count','mean','std','min','25%','50%','75%','max'
    skew       = returns.skew()
    kurt       = returns.kurtosis()
    cv         = returns.std()/returns.mean() if returns.mean()!=0 else np.nan
    var5       = returns.quantile(0.05)
    var1       = returns.quantile(0.01)
    cvar5      = returns[returns<=var5].mean()
    cvar1      = returns[returns<=var1].mean()
    cummax     = prices.cummax()
    drawdowns  = prices/cummax - 1
    max_dd     = drawdowns.min()
    # drawdown duration
    is_dd      = drawdowns < 0
    durations, count = [], 0
    for flag in is_dd:
        if flag:
            count += 1
        elif count:
            durations.append(count)
            count = 0
    if count:
        durations.append(count)
    max_dd_dur = max(durations) if durations else 0
    auto1      = returns.autocorr(lag=1)
    sharpe     = (returns.mean()/returns.std())*math.sqrt(252) if returns.std()!=0 else np.nan
    vol_ret    = volumes.pct_change().dropna() if not volumes.empty else pd.Series(dtype=float)
    turnover_corr = returns.corr(vol_ret) if not vol_ret.empty else np.nan
    mode_price    = prices.mode().iloc[0]
    mad           = returns.abs().mean()

    # Build metrics dict from raw_desc
    metrics = {
        'Count':               raw_desc['count'],
        'Mean':                raw_desc['mean'],
        'Std Dev':             raw_desc['std'],
        'Min':                 raw_desc['min'],
        '25th %ile':           raw_desc['25%'],
        'Median':              raw_desc['50%'],
        '75th %ile':           raw_desc['75%'],
        'Max':                 raw_desc['max'],
        'Skewness':            skew,
        'Kurtosis':            kurt,
        'Coeff of Var':        cv,
        'VaR 5%':              var5,
        'VaR 1%':              var1,
        'CVaR 5%':             cvar5,
        'CVaR 1%':             cvar1,
        'Max Drawdown':        max_dd,
        'Max DD Duration':     max_dd_dur,
        'Autocorr (1d)':       auto1,
        'Sharpe Ratio':        sharpe,
        'Turnover Corr':       turnover_corr,
        'Mode Price':          mode_price,
        'MAD (Returns)':       mad
    }

    # Convert to DataFrame, drop NaNs, round to ints
    stats_df = pd.Series(metrics, name='Value').to_frame()
    stats_df = stats_df.dropna()
    stats_df['Value'] = stats_df['Value'].round(2)

    st.table(stats_df)

    # 5) Price distribution histogram
    st.subheader("Price Distribution Histogram")
    fig_price_dist = px.histogram(
        prices.reset_index(drop=True),
        nbins=200,
        labels={'value':'Price'},
        title='Price Distribution'
    )
    st.plotly_chart(fig_price_dist, use_container_width=True)
