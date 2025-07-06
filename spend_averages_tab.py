# spend_averages_tab.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def run_spend_averages_tab(
    initial_price: float,
    mc_drift: float,
    mc_vol: float,
    mc_horiz: int,
    mc_sims: int,
    total_shares: float = 100
):
    """
    Tab: Compare arithmetic vs harmonic average of daily execution price
         under a fixed-notional (constant USD per day) strategy.
    """
    st.markdown("## 📊 Fixed-Notional Strategy: Price Averages Comparison")

    # --- Description ---
    st.markdown("In the **fixed-notional** strategy, you spend a constant USD amount each day.")
    
    st.markdown("If your total USD budget is \\(D\\) over a horizon of \\(H\\) days, " +
                "then on each day \\(t\\) you purchase")
    st.latex(r"""
      \Delta S_t \;=\; \frac{D/H}{p_t}
      \quad\Longrightarrow\quad
      \text{Daily spend }D_t = p_t \times \Delta S_t = \frac{D}{H}
    """)
    
    st.markdown("Thus your **per-day execution price** is simply \\(p_t\\).  We compare two averages of \\(p_t\\):")

    st.markdown("1. **Arithmetic mean price**")
    st.latex(r"""
      \overline{p}_{\rm arith}
      = \frac{1}{H}\sum_{t=1}^{H} p_t
    """)

    st.markdown("2. **Harmonic mean price**")
    st.latex(r"""
      \overline{p}_{\rm harm}
      = \left(\frac{1}{H}\sum_{t=1}^{H} \frac{1}{p_t}\right)^{-1}
    """)

    st.markdown("""
    Intuitively, the harmonic mean weights lower prices more heavily,  
    so if prices fluctuate,
    """)
    st.latex(r"\overline{p}_{\text{harm}} < \overline{p}_{\text{arith}}")
    # --- Basis-Point Comparison ---
    st.markdown("### Basis-Point Difference Between Averages")
    st.latex(r"""
    \Delta_{\mathrm{bps}} =
    \frac{\overline{p}_{\mathrm{arith}} - \overline{p}_{\mathrm{harm}}}
         {\overline{p}_{\mathrm{arith}}}
    \times 10{,}000
    """)
    
    st.markdown("""
    - A **positive** $\\Delta_{\\mathrm{bps}}$ means the harmonic average price  
      is lower (you “outperformed” the arithmetic mean by that many bps).  
    - A **negative** $\\Delta_{\\mathrm{bps}}$ means the harmonic average is higher  
      (you “underperformed” by that many bps).
    """)
    
    # --- Input summary ---
    st.markdown("---")
    st.markdown(
        f"**Inputs:**  S₀ = ${initial_price:.2f}  |  "
        f"Drift = {mc_drift:.1%}  |  "
        f"Vol = {mc_vol:.1%}  |  "
        f"Horizon = {mc_horiz} days  |  "
        f"Sims = {mc_sims:,}  |  "
        f"Total Shares = {total_shares:,}"
    )
    st.markdown("---")

    # --- 1) Simulate GBM price paths ---
    mu_d, sigma_d = mc_drift / 252.0, mc_vol / np.sqrt(252.0)
    Z    = np.random.normal(size=(mc_horiz, mc_sims))
    logs = np.vstack([
        np.zeros(mc_sims),
        np.cumsum((mu_d - 0.5 * sigma_d**2) + sigma_d * Z, axis=0)
    ])
    price_paths = initial_price * np.exp(logs)

    # --- 2) Compute per-day execution prices and their averages ---
    daily_prices = price_paths[1:]                # shape (H, sims)
    arith_price  = daily_prices.mean(axis=0)      # arithmetic mean per sim
    harm_price   = mc_horiz / (1.0 / daily_prices).sum(axis=0)

    # --- 3) Compute basis-point differences ---
    delta_bps = (arith_price - harm_price) / arith_price * 1e4

    # --- 4) Display metrics and histogram ---
    st.subheader("Mean Basis-Point Difference")
    st.metric("Δ (bps)", f"{delta_bps.mean():.1f}", f"σ = {delta_bps.std():.1f}")

    df_hist = pd.DataFrame({"Δ in bps": delta_bps})
    fig = px.histogram(
        df_hist,
        x="Δ in bps",
        nbins=200,
        title="Distribution of Basis-Point Differences"
    )
    fig.update_layout(xaxis_title="Δ (bps)", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

    # --- 5) (Optional) show sample price paths ---
    st.subheader("Sample GBM Price Paths")
    st.markdown("Below are up to 100 simulated price trajectories:")
    max_plot = min(mc_sims, 100)
    df_paths = (
        pd.DataFrame(price_paths[:, :max_plot])
          .reset_index()
          .melt(id_vars="index", var_name="Simulation", value_name="Price")
          .rename(columns={"index": "Day"})
    )
    fig2 = px.line(
        df_paths,
        x="Day",
        y="Price",
        color="Simulation",
        line_group="Simulation",
        title=f"GBM Paths (showing {max_plot} of {mc_sims:,})"
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
