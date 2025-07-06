import pandas as pd
import numpy as np
from core.benchmarks import vwap, twap, harmonic_mean

def test_twap():
    prices = pd.Series([100, 102, 98, 101, 99])
    assert round(twap(prices), 2) == 100.0

def test_vwap():
    prices = pd.Series([100, 102, 98, 101, 99])
    volumes = pd.Series([200, 100, 100, 300, 300])
    result = vwap(prices, volumes)
    assert isinstance(result, float)
    assert 99 <= result <= 101

def test_harmonic_mean():
    prices = pd.Series([100, 102, 98, 101, 99])
    result = harmonic_mean(prices)
    assert isinstance(result, float)
    assert 98 <= result <= 102
