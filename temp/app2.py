import math
import numpy as np
import pandas as pd
import streamlit as st

# Set page layout for better view
st.set_page_config(page_title="Share Buyback Pre-Trade Tool", layout="wide")

# Title of the app
st.title("Pre-Trade Share Buyback Tool")

# Sidebar inputs for data source and parameters
st.sidebar.header("1. Load Price Data")
data_source = st.sidebar.radio("Data Source:", ["Yahoo Finance", "Upload CSV", "Synthetic (GBM)"])

# Inputs for Yahoo Finance source
ticker = None
start_date = None
end_date = None
uploaded_file = None

if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Ticker Symbol (Yahoo)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with Date, Close, Volume", type=["csv"])
elif data_source == "Synthetic (GBM)":
    # If synthetic selected, we will use user parameters for generation
    # We will still use date range to determine number of days
    start_date = st.sidebar.date_input("Start Date (for synthetic timeline)", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date (for synthetic timeline)", value=pd.to_datetime("2023-12-31"))

# Synthetic data parameters (expander for clarity)
synthetic_expander = st.sidebar.expander("Synthetic Data Parameters (GBM)", expanded=(data_source=="Synthetic (GBM)"))
with synthetic_expander:
    initial_price = st.number_input("Initial Price", min_value=0.01, value=100.0, help="Starting price for synthetic data")
    gbm_drift = st.number_input("Drift (annualized, %)", value=0.0, step=0.1, help="Expected annual drift (percentage). E.g., 5% = 0.05", format="%.2f") / 100.0
    gbm_volatility = st.number_input("Volatility (annualized, %)", value=20.0, step=0.1, help="Annualized volatility (percentage).", format="%.2f") / 100.0
    avg_volume = st.number_input("Average Daily Volume (for synthetic data)", value=1000000, step=100000, help="Approximate average daily volume for synthetic data")

# Strategy execution parameters
st.sidebar.header("2. Execution Strategy Parameters")
total_shares = st.sidebar.number_input("Total Shares to Buy (TWAP)", min_value=1, value=10000, step=1000, help="Total number of shares to repurchase over the period for TWAP strategy")
participation_pct = st.sidebar.number_input("Volume Participation (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, help="Percentage of daily volume to buy each day (Volume Participation strategy)")
participation_frac = participation_pct / 100.0  # convert to fraction for calculations

# Monte Carlo simulation parameters
st.sidebar.header("3. Monte Carlo Simulation Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations", min_value=1, max_value=10000, value=500, step=100)
mc_drift = st.sidebar.number_input("Simulation Drift (annualized %)", value=0.0, step=0.1, help="Drift for Monte Carlo price paths (annual %)", format="%.2f") / 100.0
mc_volatility = st.sidebar.number_input("Simulation Volatility (annualized %)", value=20.0, step=0.1, help="Volatility for Monte Carlo paths (annual %)", format="%.2f") / 100.0
mc_horizon = st.sidebar.number_input("Simulation Horizon (days)", min_value=1, max_value=2520, value=252, step=10, help="Number of trading days to simulate into the future")

# Load or generate data when button is clicked
data_button_label = "Load Data" if data_source != "Synthetic (GBM)" else "Generate Data"
if st.sidebar.button(data_button_label):
    df = None
    error_message = None
    try:
        if data_source == "Yahoo Finance":
            # Attempt to download data using yfinance (Yahoo Finance)
            import yfinance as yf
            if ticker:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df is None or df.empty:
                    error_message = f"No data for {ticker} in range {start_date} to {end_date}"
            else:
                error_message = "No ticker provided."
            if df is not None and not df.empty:
                # Reset index to get Date column and keep only needed columns
                df = df.reset_index()
                # Ensure columns 'Close' and 'Volume' exist
                if 'Adj Close' in df.columns:
                    # Use Adjusted Close if available for more accurate historical pricing
                    df['Close'] = df['Adj Close']
                required_cols = {'Date', 'Close', 'Volume'}
                if not required_cols.issubset(df.columns):
                    error_message = "Yahoo data missing required columns."
                else:
                    df = df[list(required_cols)]  # select only Date, Close, Volume
        elif data_source == "Upload CSV":
            if uploaded_file is None:
                error_message = "No file uploaded."
            else:
                # Read CSV file into DataFrame
                df = pd.read_csv(uploaded_file)
                # Try to parse dates
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                else:
                    error_message = "Uploaded CSV must have a 'Date' column."
                # Ensure it has Close and Volume
                # Try common variations of column names
                if 'Close' not in df.columns and 'Price' in df.columns:
                    df['Close'] = df['Price']
                if 'Volume' not in df.columns and 'Vol' in df.columns:
                    df['Volume'] = df['Vol']
                if 'Close' not in df.columns or 'Volume' not in df.columns:
                    error_message = "Uploaded CSV must have 'Close' (or 'Price') and 'Volume' columns."
                # If no error so far, sort by date just in case and trim to range if given
                if error_message is None:
                    df = df.sort_values('Date').reset_index(drop=True)
                    # If user provided start/end date inputs before (though hidden in CSV mode), optionally filter by them
                    if start_date and end_date:
                        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
                        df = df.reset_index(drop=True)
        elif data_source == "Synthetic (GBM)":
            # Determine number of trading days between start and end (inclusive)
            if start_date is None or end_date is None:
                error_message = "Please provide start and end dates for synthetic data."
            else:
                # Generate business-day date range (Monday-Friday) between start and end
                bdays = pd.bdate_range(start=start_date, end=end_date)
                n_days = len(bdays)
                if n_days <= 0:
                    error_message = f"No trading days between {start_date} and {end_date}."
                else:
                    # Generate synthetic price & volume data
                    mu = gbm_drift
                    sigma = gbm_volatility
                    # Convert annual params to daily for simulation
                    mu_daily = mu / 252.0
                    sigma_daily = sigma / math.sqrt(252.0)
                    prices = [initial_price]
                    # Simulate price path
                    for _ in range(n_days - 1):
                        # Geometric Brownian Motion: S_{t+1} = S_t * exp((mu - 0.5*sigma^2) + sigma * Z)
                        rand = np.random.normal()
                        price_next = prices[-1] * math.exp((mu_daily - 0.5 * sigma_daily**2) + sigma_daily * rand)
                        prices.append(price_next)
                    # Simulate volumes (around avg_volume with some noise)
                    vol_array = np.full(n_days, avg_volume, dtype=float)
                    # Apply some random log-normal noise to volume for variability
                    vol_noise = np.random.normal(loc=0.0, scale=0.1, size=n_days)
                    vol_array = vol_array * np.exp(vol_noise)
                    vol_array = np.round(vol_array).astype(int)
                    # Build DataFrame
                    df = pd.DataFrame({"Date": bdays, "Close": prices, "Volume": vol_array})
    except Exception as e:
        # Catch any unexpected exception during data load
        error_message = str(e)

    if error_message:
        st.error(f"Failed to fetch data from {data_source}: {error_message}")
        # Fallback to synthetic data if Yahoo Finance failed
        if data_source == "Yahoo Finance":
            st.warning("Falling back to synthetic data due to data load failure.")
            # Try to generate synthetic data for the same period using provided GBM params
            try:
                bdays = pd.bdate_range(start=start_date, end=end_date)
                n_days = len(bdays)
                if n_days <= 0:
                    st.error("Cannot generate synthetic data: no trading days in specified range.")
                else:
                    # Use same synthetic generation logic as above
                    mu = gbm_drift
                    sigma = gbm_volatility
                    mu_daily = mu / 252.0
                    sigma_daily = sigma / math.sqrt(252.0)
                    prices = [initial_price]
                    for _ in range(n_days - 1):
                        rand = np.random.normal()
                        price_next = prices[-1] * math.exp((mu_daily - 0.5 * sigma_daily**2) + sigma_daily * rand)
                        prices.append(price_next)
                    vol_array = np.full(n_days, avg_volume, dtype=float)
                    vol_noise = np.random.normal(loc=0.0, scale=0.1, size=n_days)
                    vol_array = vol_array * np.exp(vol_noise)
                    vol_array = np.round(vol_array).astype(int)
                    df = pd.DataFrame({"Date": bdays, "Close": prices, "Volume": vol_array})
                    st.info("Synthetic data generated successfully for the specified period.")
            except Exception as e:
                st.error(f"Synthetic data generation failed: {e}")
                    # If synthetic generation also fails (very unlikely), df stays None.
    # Store the dataframe in session state if loaded
    if 'df' in st.session_state:
        st.session_state.pop('df')  # clear old data if any
    if df is not None and not df.empty:
        st.session_state['df'] = df

# Proceed if we have data loaded in session state
if 'df' in st.session_state:
    data_df = st.session_state['df']
    # Ensure DataFrame has proper types
    if 'Date' in data_df.columns:
        data_df['Date'] = pd.to_datetime(data_df['Date'])
    # Sort by date and reset index
    data_df = data_df.sort_values('Date').reset_index(drop=True)
    # Display basic info
    st.subheader("Loaded Price Data")
    # Show date range and number of days loaded
    start_str = data_df['Date'].iloc[0].strftime("%Y-%m-%d")
    end_str = data_df['Date'].iloc[-1].strftime("%Y-%m-%d")
    st.write(f"**Data Period:** {start_str} to {end_str}  &nbsp; | &nbsp; **Trading Days:** {len(data_df)}")
    # Show a quick line chart of price data for context
    st.line_chart(data_df.set_index('Date')['Close'], height=250)

    # Calculate benchmark metrics
    prices = data_df['Close'].to_numpy()
    volumes = data_df['Volume'].to_numpy() if 'Volume' in data_df.columns else None
    vwap = None
    if volumes is not None:
        total_vol = np.nansum(volumes)
        if total_vol > 0:
            vwap = np.nansum(prices * volumes) / total_vol
    twap_price = np.nanmean(prices)
    harmonic_mean_price = len(prices) / np.nansum(1.0 / prices) if len(prices) > 0 else None

    # Display benchmark metrics
    st.markdown("**Benchmark Price Metrics:**")
    metrics_cols = st.columns(3)
    metrics_cols[0].metric(label="Period VWAP", value=f"{vwap:.3f}" if vwap is not None else "N/A")
    metrics_cols[1].metric(label="Period TWAP (avg price)", value=f"{twap_price:.3f}" if twap_price is not None else "N/A")
    metrics_cols[2].metric(label="Harmonic Mean Price", value=f"{harmonic_mean_price:.3f}" if harmonic_mean_price is not None else "N/A")

    # Prepare execution strategy logs
    num_days = len(data_df)
    # TWAP strategy: equal shares per day
    shares_per_day = total_shares / num_days
    twap_records = []
    for idx, row in data_df.iterrows():
        price = row['Close']
        # Shares (TWAP) - using equal shares each trading day
        shares = shares_per_day
        cost = shares * price
        twap_records.append({"Date": row['Date'].strftime("%Y-%m-%d"), "Price": price, "Shares Bought": shares, "Cost": cost})
    twap_df = pd.DataFrame(twap_records)
    total_cost_twap = twap_df["Cost"].sum()
    avg_price_twap_exec = total_cost_twap / total_shares  # average execution price for TWAP strategy

    # Volume Participation strategy: fixed percentage of daily volume
    vp_records = []
    total_shares_vp = 0.0
    for idx, row in data_df.iterrows():
        price = row['Close']
        vol = row['Volume'] if 'Volume' in data_df.columns else 0
        shares = participation_frac * vol  # shares bought that day
        cost = shares * price
        total_shares_vp += shares
        vp_records.append({"Date": row['Date'].strftime("%Y-%m-%d"), "Price": price, "Volume": vol, "Shares Bought": shares, "Cost": cost})
    vp_df = pd.DataFrame(vp_records)
    total_cost_vp = vp_df["Cost"].sum()
    avg_price_vp_exec = total_cost_vp / total_shares_vp if total_shares_vp != 0 else None

    # Use tabs to separate deterministic vs Monte Carlo results
    tab1, tab2 = st.tabs(["Deterministic Simulation", "Monte Carlo Simulation"])

    with tab1:
        st.subheader("Deterministic Execution Simulation")
        # Provide summary of total costs and average execution prices for the strategies
        st.write(f"**TWAP Strategy:** Total Cost = {total_cost_twap:,.2f}, Average Execution Price = {avg_price_twap_exec:.3f}")
        st.write(f"**Volume Participation Strategy:** Total Cost = {total_cost_vp:,.2f}, Total Shares = {total_shares_vp:,.0f}, Average Execution Price = {avg_price_vp_exec:.3f}" if avg_price_vp_exec is not None else 
                 f"**Volume Participation Strategy:** Total Cost = {total_cost_vp:,.2f}, Total Shares = {total_shares_vp:,.0f}")
        # Show execution logs as expandable dataframes
        st.write("**Daily Execution Log – TWAP Strategy**")
        st.dataframe(twap_df, use_container_width=True)
        csv_twap = twap_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download TWAP Log (CSV)", data=csv_twap, file_name="TWAP_execution_log.csv", mime="text/csv")

        st.write("**Daily Execution Log – Volume Participation Strategy**")
        st.dataframe(vp_df, use_container_width=True)
        csv_vp = vp_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download VP Log (CSV)", data=csv_vp, file_name="VolumeParticipation_execution_log.csv", mime="text/csv")

    with tab2:
        st.subheader("Monte Carlo Simulation of Future Costs")
        # Get baseline values for simulation
        current_price = data_df['Close'].iloc[-1]
        # Use average volume from the loaded data as baseline future daily volume (assuming similar market activity)
        baseline_daily_vol = data_df['Volume'].mean() if 'Volume' in data_df.columns else avg_volume

        # Monte Carlo price simulation (Geometric Brownian Motion)
        n = int(num_simulations)
        H = int(mc_horizon)
        mu = mc_drift
        sigma = mc_volatility
        if H <= 0 or n <= 0:
            st.error("Horizon days and number of simulation must be positive.")
        else:
            # Convert annual drift/vol to daily
            mu_daily = mu / 252.0
            sigma_daily = sigma / math.sqrt(252.0)
            # Simulate log returns for H days for n simulation
            # Shape for random normals: (H, n)
            rand_normals = np.random.normal(size=(H, n))
            # Calculate daily log-return increments: (mu - 0.5*sigma^2) + sigma * Z
            daily_increments = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * rand_normals
            # Cumulative log returns for each simulation
            cum_log_returns = np.cumsum(daily_increments, axis=0)
            # Prepend a row of zeros for initial price (to represent day0)
            zeros = np.zeros((1, n))
            cum_log_returns = np.vstack([zeros, cum_log_returns])
            # Price paths: each column is a simulation, each row a day (including day0)
            price_paths = current_price * np.exp(cum_log_returns)
            # price_paths has shape (H+1, n), where row 0 is initial price, rows 1..H are simulated days

            # Compute total costs for each simulation for both strategies
            # TWAP: buy total_shares evenly over H days -> shares per day
            shares_per_day_MC = total_shares / H
            # Volume participation: shares per day = fraction * baseline_volume each day
            shares_per_day_vp_MC = participation_frac * baseline_daily_vol

            # Sum of prices over each path (excluding the initial price at row 0 if we consider trades only on H days)
            # If we assume trading happens on each of the H days forward (not including day0), we sum rows 1..H.
            price_sum_per_sim = price_paths[1:].sum(axis=0)  # shape (n,)
            # Total cost arrays
            total_costs_twap = shares_per_day_MC * price_sum_per_sim
            total_costs_vp = shares_per_day_vp_MC * price_sum_per_sim

            # Prepare data for histogram (as pandas series for convenience)
            twap_cost_series = pd.Series(total_costs_twap, name="TWAP Total Cost")
            vp_cost_series = pd.Series(total_costs_vp, name="VP Total Cost")

            # Plot histograms for cost distributions
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(twap_cost_series, bins=30, color='#4c78a8', edgecolor='black')
            axes[0].set_title("TWAP Cost Distribution")
            axes[0].set_xlabel("Total Cost")
            axes[0].set_ylabel("Frequency")
            axes[1].hist(vp_cost_series, bins=30, color='#f58518', edgecolor='black')
            axes[1].set_title("Volume Participation Cost Distribution")
            axes[1].set_xlabel("Total Cost")
            axes[1].set_ylabel("Frequency")
            plt.tight_layout()
            st.pyplot(fig)

            # Display basic statistics of the distributions
            mean_twap = twap_cost_series.mean()
            mean_vp = vp_cost_series.mean()
            std_twap = twap_cost_series.std()
            std_vp = vp_cost_series.std()
            st.write(f"**TWAP Cost:** Mean = {mean_twap:,.2f}, Std Dev = {std_twap:,.2f}")
            st.write(f"**VP Cost:** Mean = {mean_vp:,.2f}, Std Dev = {std_vp:,.2f}")

            # Offer download of simulation results
            sim_results_df = pd.DataFrame({"TWAP_Total_Cost": total_costs_twap, "VP_Total_Cost": total_costs_vp})
            csv_sim = sim_results_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Simulation Cost Distribution (CSV)", data=csv_sim,
                               file_name="monte_carlo_costs.csv", mime="text/csv")
