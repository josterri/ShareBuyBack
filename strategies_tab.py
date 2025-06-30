# strategies_tab.py
import streamlit as st
from strategies import simulate_twap, simulate_volume_participation

def run_execution_tab(df, total_shares, participation_frac):
    """
    Renders the Execution Strategies tab:
     - TWAP strategy log + download
     - Volume Participation strategy log + download
    """
    # TWAP
    st.subheader("TWAP Strategy")
    twap_df = simulate_twap(df, total_shares)
    st.dataframe(twap_df, use_container_width=True)
    st.download_button(
        label="Download TWAP CSV",
        data=twap_df.to_csv(index=False),
        file_name="twap_log.csv",
        mime="text/csv"
    )

    # Volume Participation
    st.subheader("Volume Participation Strategy")
    vp_df = simulate_volume_participation(df, participation_frac)
    st.dataframe(vp_df, use_container_width=True)
    st.download_button(
        label="Download VP CSV",
        data=vp_df.to_csv(index=False),
        file_name="vp_log.csv",
        mime="text/csv"
    )
