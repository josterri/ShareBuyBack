import pytest
import numpy as np
from core.simulator import simulate_execution

def test_uniform_execution_price():
    prices = [100, 102, 98, 101, 99]
    total_value = 50000
    days = 5
    avg_price = simulate_execution(prices, total_value, days)
    
    # Check that average price is within the price range
    assert 98 <= avg_price <= 102

def test_execution_insufficient_data():
    prices = [100, 102]
    total_value = 10000
    days = 5
    with pytest.raises(ValueError):
        simulate_execution(prices, total_value, days)

def test_invalid_strategy():
    prices = [100] * 10
    total_value = 10000
    days = 5
    with pytest.raises(NotImplementedError):
        simulate_execution(prices, total_value, days, strategy="backload")
