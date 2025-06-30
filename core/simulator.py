import numpy as np

def simulate_execution(prices, total_value, days, strategy="uniform"):
    """
    Simulate average execution price over a period.

    Parameters:
        prices (array-like): historical prices
        total_value (float): total capital to invest
        days (int): number of trading days to execute
        strategy (str): execution strategy ("uniform" only supported now)

    Returns:
        float: average execution price
    """
    prices = np.array(prices[:days])

    if len(prices) < days:
        raise ValueError("Not enough price data for the given execution window.")

    if strategy == "uniform":
        daily_value = total_value / days
        shares_per_day = daily_value / prices
        total_shares = np.sum(shares_per_day)
        total_cost = np.sum(shares_per_day * prices)
        return total_cost / total_shares
    else:
        raise NotImplementedError(f"Strategy '{strategy}' not implemented.")
