# descriptive.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def run_descriptive_analysis(df: pd.DataFrame):
    """
    Performs and renders a full descriptive analysis of df['Close']:
     1. Price time series
     2. Histogram of daily returns
     3. Rolling 20-day moving average & volatility
     4. Summary statistics
     5. Price distribution histogram
    """
    # Prepare and sort
    df2 = df.copy()
    if 'Date' in df2.columns:
        df2['Date'] = pd.to_datetime(df2['Date'])
    df2.sort_values('Date', inplace=True)
    
    prices = df2['Close']
    returns = prices.pct_change().dropna()
    volumes = df2.get('Volume', pd.Series(dtype=float))

    # 1) Price time series
    st.subheader("Price Time Series")
    st.line_chart(df2.set_index('Date')['Close'])

    # 2) Histogram of daily returns
    st.subheader("Daily Returns Distribution")
    fig_ret, ax_ret = plt.subplots(figsize=(8, 4))
    ax_ret.hist(returns, bins=50, edgecolor='black')
    ax_ret.set_xlabel("Daily Return")
    ax_ret.set_ylabel("Frequency")
    st.pyplot(fig_ret)

    # 3) Rolling statistics
    st.subheader("20-Day Rolling Average & Volatility")
    rolling_mean = prices.rolling(window=20).mean()
    rolling_vol  = returns.rolling(window=20).std() * math.sqrt(252)
    dates_ret    = df2['Date'].iloc[1:]

    fig_roll, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(df2['Date'], rolling_mean, label='20-day MA')
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[1].plot(dates_ret, rolling_vol, label='20-day Vol (ann.)', color='orange')
    axes[1].set_ylabel("Volatility")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig_roll)

    # 4) Summary statistics
    st.subheader("Summary Statistics")
    price_desc = prices.describe()
    return_desc = returns.describe()
    skew       = returns.skew()
    kurt       = returns.kurtosis()
    cv         = returns.std() / returns.mean() if returns.mean() != 0 else np.nan
    var5       = returns.quantile(0.05)
    var1       = returns.quantile(0.01)
    cvar5      = returns[returns <= var5].mean()
    cvar1      = returns[returns <= var1].mean()
    cummax     = prices.cummax()
    drawdowns  = prices / cummax - 1
    max_dd     = drawdowns.min()
    # Drawdown duration
    is_dd      = drawdowns < 0
    durations  = []
    count      = 0
    for flag in is_dd:
        if flag:
            count += 1
        else:
            if count:
                durations.append(count)
            count = 0
    if count:
        durations.append(count)
    max_dd_dur = max(durations) if durations else 0
    auto1      = returns.autocorr(lag=1)
    sharpe     = (returns.mean() / returns.std()) * math.sqrt(252) if returns.std() != 0 else np.nan
    vol_ret    = volumes.pct_change().dropna() if not volumes.empty else pd.Series(dtype=float)
    turnover_corr = returns.corr(vol_ret) if not vol_ret.empty else np.nan
    mode_price    = prices.mode().iloc[0]
    mad           = returns.abs().mean()  # mean absolute deviation

    summary = pd.DataFrame({
        'Metric': [
            'Count', 'Mean', 'Std Dev', 'Min', '25th %ile', 'Median',
            '75th %ile', 'Max', 'Skewness', 'Kurtosis', 'Coeff of Var',
            'VaR 5%', 'VaR 1%', 'CVaR 5%', 'CVaR 1%', 'Max Drawdown',
            'Max Drawdown Duration', 'Autocorr (lag1)', 'Sharpe Ratio',
            'Turnover Corr', 'Mode Price', 'MAD (Returns)'
        ],
        'Value': [
            price_desc['count'], price_desc['mean'], price_desc['std'],
            price_desc['min'], price_desc['25%'], price_desc['50%'],
            price_desc['75%'], price_desc['max'], skew, kurt, cv,
            var5, var1, cvar5, cvar1, max_dd, max_dd_dur,
            auto1, sharpe, turnover_corr, mode_price, mad
        ]
    }).set_index('Metric')

    st.table(summary)

    # 5) Price distribution histogram
    st.subheader("Price Distribution Histogram")
    fig_price, ax_price = plt.subplots(figsize=(8, 4))
    ax_price.hist(prices, bins=30, edgecolor='black')
    ax_price.set_xlabel("Price")
    ax_price.set_ylabel("Frequency")
    st.pyplot(fig_price)
