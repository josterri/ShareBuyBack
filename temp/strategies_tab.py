# strategies_tab.py
import streamlit as st
import pandas as pd
from temp.strategies import simulate_twap, simulate_volume_participation

def run_execution_tab(df: pd.DataFrame, total_shares: float, participation_frac: float):
    """
    Renders the Execution Strategies tab with:
     - Date formatted as YYYY-MM-DD (no time)
     - All numeric columns rounded to integers (no decimal digits)
    """
    # Prepare
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Date'])

    # --- TWAP Strategy ---
    st.subheader("TWAP Strategy")
    twap_df = simulate_twap(df2, total_shares)
    # Format Date
    twap_df['Date'] = pd.to_datetime(twap_df['Date']).dt.strftime('%Y-%m-%d')
    # Round numeric columns to integers
    for col in ['Price', 'Shares', 'Cost']:
        twap_df[col] = twap_df[col].round(0).astype(int)
    # Reorder and display
    twap_df = twap_df[['Date', 'Price', 'Shares', 'Cost']]
    st.dataframe(twap_df, use_container_width=True)
    st.download_button(
        label="Download TWAP CSV",
        data=twap_df.to_csv(index=False),
        file_name="twap_log.csv",
        mime="text/csv"
    )

    # --- Volume Participation Strategy ---
    st.subheader("Volume Participation Strategy")
    vp_df = simulate_volume_participation(df2, participation_frac)
    vp_df['Date'] = pd.to_datetime(vp_df['Date']).dt.strftime('%Y-%m-%d')
    for col in ['Price', 'Shares', 'Cost']:
        vp_df[col] = vp_df[col].round(0).astype(int)
    vp_df = vp_df[['Date', 'Price', 'Shares', 'Cost']]
    st.dataframe(vp_df, use_container_width=True)
    st.download_button(
        label="Download VP CSV",
        data=vp_df.to_csv(index=False),
        file_name="vp_log.csv",
        mime="text/csv"
    )
