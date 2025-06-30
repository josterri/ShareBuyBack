# descriptive.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run_descriptive_analysis(df: pd.DataFrame):
    """
    Performs and renders a full descriptive analysis of df['Close']:
     - Summary statistics table
     - Histogram of price distribution
     - Time series line chart of prices
    """
    # Ensure Date column is datetime
    df = df.copy()  
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    prices = df['Close']

    # 1) Summary statistics
    stats = prices.describe().rename({
        'count': 'Count',
        'mean': 'Mean',
        'std': 'Std Dev',
        'min': 'Min',
        '25%': '25th %ile',
        '50%': 'Median',
        '75%': '75th %ile',
        'max': 'Max'
    })
    st.subheader("Full Descriptive Statistics")
    st.table(stats.to_frame("Price"))

    # 2) Histogram
    st.subheader("Price Distribution Histogram")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(prices, bins=30, edgecolor='black')
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig)

    # 3) Time series chart
    st.subheader("Time Series of Price")
    st.line_chart(df.set_index('Date')['Close'], height=300)
