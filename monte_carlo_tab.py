import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Cache folder
CACHE_DIR = os.path.join(os.path.dirname(__file__), "mc_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _get_cache_path(
    S0: float,
    mu: float,
    sigma: float,
    horizon: int,
    sims: int,
    total_shares: float
) -> str:
    # Human-readable filename based only on your inputs
    fname = (
        f"mc_S0_{S0:.2f}"
        f"_mu_{mu:.4f}"
        f"_sigma_{sigma:.4f}"
        f"_h{horizon}"
        f"_s{sims}"
        f"_n{int(total_shares)}"
        ".csv"
    )
    return os.path.join(CACHE_DIR, fname)

def load_or_run_simulation(
    S0: float,
    mu: float,
    sigma: float,
    horizon: int,
    sims: int,
    total_shares: float
) -> pd.DataFrame:
    cache_path = _get_cache_path(S0, mu, sigma, horizon, sims, total_shares)
    st.write(f"🔍 Looking for cache at `{cache_path}`")

    if os.path.exists(cache_path):
        st.success(f"✅ Loaded cached MC results ({os.path.basename(cache_path)})")
        return pd.read_csv(cache_path)

    st.warning("🔄 No cache hit; running Monte Carlo…")
    # 1) Simulate price paths via GBM
    mu_d    = mu / 252.0
    sigma_d = sigma / np.sqrt(252.0)
    Z       = np.random.normal(size=(horizon, sims))
    logs    = np.vstack([np.zeros(sims),
                         np.cumsum((mu_d - 0.5*sigma_d**2) + sigma_d * Z, axis=0)])
    price_paths = S0 * np.exp(logs)

    # 2) Compute cost for TWAP vs Fixed-Notional
    #    - TWAP cost = fixed shares at each day's price
    #    - USD cost   = fixed notional each day
    daily_notional = total_shares * S0 / horizon
    shares_usd     = (daily_notional / price_paths[1:]).sum(axis=0)
    twap_cost      = (price_paths[1:].mean(axis=0) * total_shares)
    usd_cost       = (daily_notional * horizon)  # = total_notional
    df = pd.DataFrame({
        "TWAP Cost": twap_cost,
        "USD Cost": usd_cost
    })

    df.to_csv(cache_path, index=False)
    st.success(f"💾 Saved new results to cache ({os.path.basename(cache_path)})")
    return df

def run_monte_carlo_tab(
    df: pd.DataFrame,
    initial_price: float,
    mc_drift: float,
    mc_vol: float,
    mc_horiz: int,
    mc_sims: int,
    total_shares: float
):
    # Load from cache or re-run
    costs_df = load_or_run_simulation(
        initial_price, mc_drift, mc_vol, mc_horiz, mc_sims, total_shares
    )

    # 1) Price difference in basis points
    #    avg exec price TWAP = TWAP Cost / shares
    #    avg exec price USD  = USD Cost  / shares_usd
    P_twap = costs_df["TWAP Cost"] / total_shares
    P_usd  = costs_df["USD Cost"]  / (costs_df["USD Cost"] / initial_price / mc_horiz)  # but fixed notional buys shares_usd
    bps_diff = (P_twap - P_usd) / P_twap * 1e4

    st.subheader("1. Price Difference in Basis Points")
    df_bps = pd.DataFrame({"Price Outperformance (bps)": bps_diff})
    fig_bps = px.histogram(df_bps, x="Price Outperformance (bps)", nbins=200)
    fig_bps.update_layout(xaxis_title="bps Δ", yaxis_title="Frequency")
    st.plotly_chart(fig_bps, use_container_width=True)
    st.metric("Mean bps Δ", f"{bps_diff.mean():.1f}", f"σ={bps_diff.std():.1f}")

    # 2) Simulated GBM paths
    st.subheader("2. Simulated GBM Price Paths")
    mu_d    = mc_drift / 252.0
    sigma_d = mc_vol   / np.sqrt(252.0)
    Z       = np.random.normal(size=(mc_horiz, mc_sims))
    logs    = np.vstack([np.zeros(mc_sims),
                         np.cumsum((mu_d - 0.5*sigma_d**2) + sigma_d * Z, axis=0)])
    price_paths = initial_price * np.exp(logs)

    max_paths = min(mc_sims, 100)
    df_paths = (
        pd.DataFrame(price_paths[:, :max_paths])
          .reset_index()
          .melt(id_vars="index", var_name="sim", value_name="price")
          .rename(columns={"index": "Day"})
    )
    fig_paths = px.line(df_paths, x="Day", y="price",
                        color="sim", line_group="sim",
                        title=f"GBM Paths (showing {max_paths} of {mc_sims} sims)")
    fig_paths.update_layout(showlegend=False)
    st.plotly_chart(fig_paths, use_container_width=True)
