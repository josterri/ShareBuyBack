import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- App Configuration ---
st.set_page_config(
    page_title="Investment Strategy Simulator",
    page_icon="📊",
    layout="wide"
)

# --- Simulation Engine ---
def run_gbm_simulation(
    s0: float,
    mu: float,
    sigma: float,
    n_days: int,
    n_paths: int,
    trading_days_per_year: int = 252,
) -> np.ndarray:
    """
    Generates multiple asset price paths using Geometric Brownian Motion.
    Returns a 2D array of price paths with shape (n_paths, n_days + 1).
    """
    dt = 1 / trading_days_per_year
    # The Ito-corrected drift for the log-process
    m = mu - 0.5 * sigma**2
    
    # Generate random shocks for all paths and steps at once
    random_shocks = np.random.normal(0.0, 1.0, size=(n_paths, n_days))
    
    # Calculate daily log returns for all paths
    log_returns = (m * dt) + (sigma * np.sqrt(dt) * random_shocks)
    
    # Calculate cumulative log returns and exponentiate to get prices
    cumulative_log_returns = np.cumsum(log_returns, axis=1)
    price_paths = s0 * np.exp(cumulative_log_returns)
    
    # Prepend the initial price S0 to each path
    initial_prices = np.full((n_paths, 1), s0)
    full_price_paths = np.hstack([initial_prices, price_paths])
    
    return full_price_paths

# --- Strategy Analysis ---
def analyze_investment_strategy(
    price_paths: np.ndarray, daily_investment: float
) -> pd.DataFrame:
    """
    Analyzes the daily investment strategy across all simulated price paths.
    """
    n_paths, n_steps = price_paths.shape
    n_days = n_steps - 1
    
    # Prices used for investment (from day 1 to end)
    investment_prices = price_paths[:, 1:]
    
    # Vectorized calculation of shares acquired daily
    shares_acquired_daily = daily_investment / investment_prices
    
    # Total shares acquired per path
    total_shares_acquired = np.sum(shares_acquired_daily, axis=1)
    
    # Total USD invested is constant for all paths
    total_usd_invested = daily_investment * n_days
    
    # Calculate performance metrics per path
    avg_execution_price = total_usd_invested / total_shares_acquired
    simple_avg_price = np.mean(investment_prices, axis=1)
    performance_score = avg_execution_price / simple_avg_price
    performance_bps = (1-performance_score) * 10000  # Convert to basis points
    terminal_price = price_paths[:, -1]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'path_id': range(1, n_paths + 1),
        'terminal_price': terminal_price,
        'total_usd_invested': total_usd_invested,
        'total_shares_acquired': total_shares_acquired,
        'avg_execution_price': avg_execution_price,
        'simple_avg_price': simple_avg_price,
        'performance_bps': performance_bps,
    })
    
    return results_df

# --- Visualization Functions ---
def plot_price_paths(price_paths: np.ndarray, n_to_plot: int = 50):
    """Plots a sample of the simulated price paths."""
    n_paths, n_steps = price_paths.shape
    path_ids_to_plot = np.random.choice(range(n_paths), size=min(n_paths, n_to_plot), replace=False)
    
    df_list = []
    for path_id in path_ids_to_plot:
        df_list.append(pd.DataFrame({'Day': range(n_steps), 'Price': price_paths[path_id, :], 'Path': f'Path_{path_id+1}'}))
    df_to_plot = pd.concat(df_list)
    
    fig = px.line(
        df_to_plot, x='Day', y='Price', color='Path',
        title=f'Sample of {n_to_plot} Simulated Price Paths (The Cone of Possibility)',
        labels={'Day': 'Trading Day', 'Price': 'Asset Price (USD)'}
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_terminal_price_distribution(results_df: pd.DataFrame):
    """Plots the histogram of terminal asset prices."""
    fig = px.histogram(
        results_df, x='terminal_price', nbins=75, histnorm='probability density',
        marginal='rug', title='Distribution of Terminal Asset Prices',
        labels={'terminal_price': 'Terminal Price (USD)'}
    )
    return fig

def plot_performance_distribution(results_df: pd.DataFrame):
    """Plots the histogram of the strategy performance in basis points."""
    fig = px.histogram(
        results_df, x='performance_bps', nbins=275, histnorm='probability density',
        title='Distribution of Strategy Performance (Positive is Better)',
        labels={'performance_bps': 'Performance (bps) vs. Simple Average Price'}
    )
    fig.add_vline(
        x=0.0, line_width=2, line_dash="dash", line_color="red",
        annotation_text="Neutral (0 bps)", annotation_position="top right"
    )
    return fig

def plot_performance_vs_terminal_price(results_df: pd.DataFrame):
    """Creates a scatter plot of performance vs. terminal price."""
    fig = px.scatter(
        results_df, x='terminal_price', y='performance_bps',
        title='Strategy Performance vs. Market Outcome',
        labels={
            'terminal_price': 'Terminal Asset Price (USD)',
            'performance_bps': 'Performance (bps)'
        },
        hover_data=['path_id'],
        color='performance_bps',
        color_continuous_scale=px.colors.sequential.Viridis_r
    )
    fig.add_hline(y=0.0, line_width=2, line_dash="dash", line_color="red")
    return fig

def plot_single_path_analysis(price_path: np.ndarray, results_row: pd.Series, daily_investment: float):
    """Visualizes a single path with running performance metrics and a secondary axis for performance bps."""
    # Calculate running performance metrics
    investment_prices = price_path[1:]
    days_axis = np.arange(1, len(price_path))
    
    shares_daily = daily_investment / investment_prices
    cumulative_shares = np.cumsum(shares_daily)
    cumulative_investment = days_axis * daily_investment
    running_avg_exec_price = cumulative_investment / cumulative_shares
    
    # Use an expanding mean for the running simple average
    running_simple_avg_price = pd.Series(investment_prices).expanding().mean().to_numpy()
    
    # Calculate running performance in BPS (Negative is better)
    running_performance_bps = ((running_avg_exec_price / running_simple_avg_price) - 1) * 10000

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for the primary Y-axis (Price)
    fig.add_trace(
        go.Scatter(x=np.arange(len(price_path)), y=price_path, name="Asset Price", line=dict(color='royalblue')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=days_axis, y=running_avg_exec_price, name="Running Avg Exec Price", line=dict(color='green', dash='dash')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=days_axis, y=running_simple_avg_price, name="Running Simple Avg Price", line=dict(color='orange', dash='dash')),
        secondary_y=False,
    )
    
    # Add trace for the secondary Y-axis (Performance BPS)
    fig.add_trace(
        go.Scatter(x=days_axis, y=running_performance_bps, name="Performance (bps)", line=dict(color='firebrick', dash='dot')),
        secondary_y=True,
    )
    
    # Set titles and labels
    fig.update_layout(
        title_text=f'Deep Dive on Path #{results_row["path_id"]} (Final Performance: {results_row["performance_bps"]:.1f} bps)',
        xaxis_title="Trading Day",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="<b>Price (USD)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Performance (bps)</b>", secondary_y=True)

    return fig


def plot_cumulative_shares(price_path: np.ndarray, daily_investment: float):
    """Plots the cumulative shares acquired over a single path."""
    investment_prices = price_path[1:]
    shares_daily = daily_investment / investment_prices
    cumulative_shares = np.cumsum(shares_daily)
    days = np.arange(1, len(price_path))

    fig = px.area(x=days, y=cumulative_shares, title='Cumulative Shares Acquired Over Time (Path #1)',
                  labels={'x': 'Trading Day', 'y': 'Total Shares Acquired'})
    return fig


# --- Caching ---
@st.cache_data
def run_full_simulation(s0, mu, sigma, n_days, n_paths, daily_investment):
    """Cached function to run simulation and analysis."""
    price_paths = run_gbm_simulation(s0, mu, sigma, n_days, n_paths)
    results_df = analyze_investment_strategy(price_paths, daily_investment)
    return price_paths, results_df

# --- Main Application UI and Logic ---
st.title("📊 Advanced Share Buyback Strategy Simulator")
st.markdown("""
This application uses a **Monte Carlo simulation** based on **Geometric Brownian Motion (GBM)** to analyze a daily dollar-cost averaging (DCA) investment strategy. 
Configure the market and strategy parameters in the sidebar, then click "Run Simulation" to explore the results across thousands of potential market scenarios.
""")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    s0 = st.number_input("Initial Asset Price ($S_0$)", min_value=1.0, value=100.0, step=1.0)
    mu = st.slider("Annualized Drift (μ)", -0.5, 0.5, 0.0, 0.01, format="%.2f", help="The expected annual rate of return.")
    sigma = st.slider("Annualized Volatility (σ)", 0.05, 1.0, 0.25, 0.01, format="%.2f", help="The annual standard deviation of returns, a measure of risk.")
    
    st.header("📈 Strategy Parameters")
    n_days = st.number_input("Investment Period (Days)", min_value=10, value=125, step=5)
    daily_investment = st.number_input("Daily Investment (USD)", min_value=1.0, value=1000.0, step=10.0)
    
    st.header("🎲 Monte Carlo Parameters")
    n_paths = st.number_input("Number of Simulations (Paths)", min_value=100, value=500, step=100)

    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# --- Main Application Logic ---
if run_button:
    price_paths, results_df = run_full_simulation(s0, mu, sigma, int(n_days), int(n_paths), daily_investment)
    st.session_state.price_paths = price_paths
    st.session_state.results_df = results_df
    st.session_state.daily_investment = daily_investment

# --- Display Results ---
# This block now checks if results exist and are a valid DataFrame before trying to display them.
# This makes the app more robust and prevents KeyErrors from stale session state.
if 'results_df' in st.session_state and isinstance(st.session_state.results_df, pd.DataFrame):
    st.header("🔍 Simulation Results Analysis")

    # --- Key Metrics ---
    results = st.session_state.results_df
    perf_bps = results['performance_bps']
    favorable_outcomes = (perf_bps > 0).sum()
    favorable_percentage = (favorable_outcomes / len(results)) * 100
    p5, p95 = perf_bps.quantile([0.05, 0.95])

    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Median Performance", f"{perf_bps.median():.1f} bps", help="The 50th percentile outcome. Positive is better.")
    col2.metric("Mean Performance", f"{perf_bps.mean():.1f} bps", help="The average outcome across all simulations. Positive is better.")
    col3.metric("Favorable Outcomes", f"{favorable_percentage:.1f}%", help="The percentage of simulations where the strategy's average price was better than the simple average (Performance < 0 bps).")
    col4.metric("Std. Dev. of Performance", f"{perf_bps.std():.1f} bps", help="Measures the dispersion or 'risk' of the performance. Higher values mean more uncertainty.")
    st.markdown(f"**90% Confidence Interval for Performance:** The performance landed between `{p5:.1f}` bps and `{p95:.1f}` bps in 90% of the simulations.")


    # --- Create tabs for different views ---
    tab1, tab2, tab3, tab4 = st.tabs(["Summary & Distributions", "Single Path Deep Dive", "Aggregate Analysis", "Raw Data"])

    with tab1:
        st.markdown("""
        These charts show the statistical distribution of the two key outcomes: the **strategy's performance** and the **final asset price**.
        The performance is measured in basis points (bps) versus the simple average price. **Positive is better.**
        """)
        c1, c2 = st.columns(2)
        with c1:
            fig_perf = plot_performance_distribution(results)
            st.plotly_chart(fig_perf, use_container_width=True)
        with c2:
            fig_terminal = plot_terminal_price_distribution(results)
            st.plotly_chart(fig_terminal, use_container_width=True)

    with tab2:
        st.markdown("""
        This section provides a detailed look at the **first simulated path**. The primary y-axis shows the asset's price along with the **running average execution price** and the **running simple average price**. 
        The secondary y-axis shows the cumulative **performance in basis points** as the strategy progresses.
        """)
        first_path_prices = st.session_state.price_paths[0]
        first_path_results = results.iloc[0]
        daily_inv = st.session_state.daily_investment
        
        fig_single = plot_single_path_analysis(first_path_prices, first_path_results, daily_inv)
        st.plotly_chart(fig_single, use_container_width=True)

        fig_cum_shares = plot_cumulative_shares(first_path_prices, daily_inv)
        st.plotly_chart(fig_cum_shares, use_container_width=True)

    with tab3:
        st.markdown("""
        These visualizations analyze the results across all simulations at once. The scatter plot helps identify if the strategy's
        performance is correlated with the final market outcome (i.e., does it do better in bull or bear markets?). The line plot shows the
        "cone of possibility" for the asset's price.
        """)
        fig_scatter = plot_performance_vs_terminal_price(results)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        fig_paths = plot_price_paths(st.session_state.price_paths, n_to_plot=100)
        st.plotly_chart(fig_paths, use_container_width=True)

    with tab4:
        st.subheader("Full Simulation Results Table")
        st.dataframe(results[['path_id', 'terminal_price', 'avg_execution_price', 'simple_avg_price', 'performance_bps']].style.format({
            'terminal_price': '${:,.2f}',
            'avg_execution_price': '${:,.2f}',
            'simple_avg_price': '${:,.2f}',
            'performance_bps': '{:.1f}',
        }))
else:
    st.info("Please configure parameters in the sidebar and click 'Run Simulation' to begin.")

