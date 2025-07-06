import unittest
import pandas as pd

# Import the functions to test
from temp.metrics import calculate_vwap, calculate_twap, calculate_harmonic_mean
from temp.strategies import simulate_twap_strategy, simulate_volume_participation_strategy

class BuybackToolTests(unittest.TestCase):
    def setUp(self):
        # Set up a simple DataFrame for testing with 2 days of data
        dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
        prices = [10.0, 20.0]
        volumes = [1000, 1000]  # constant volume each day for simplicity
        self.test_data = pd.DataFrame({"Close": prices, "Volume": volumes}, index=dates)
    
    def test_vwap_calculation(self):
        # VWAP for prices [10, 20] with volumes [1000, 1000] should be 15.0
        vwap = calculate_vwap(self.test_data["Close"], self.test_data["Volume"])
        self.assertAlmostEqual(vwap, 15.0, places=5)
    
    def test_twap_calculation(self):
        # TWAP (simple average) for [10, 20] is 15.0
        twap = calculate_twap(self.test_data["Close"])
        self.assertEqual(twap, 15.0)
    
    def test_harmonic_mean_calculation(self):
        # Harmonic mean for [10, 20]: should be around 13.33
        hmean = calculate_harmonic_mean(self.test_data["Close"])
        expected = 2.0 / (1/10.0 + 1/20.0)
        self.assertAlmostEqual(hmean, expected, places=2)
    
    def test_twap_strategy_distribution(self):
        # Test TWAP strategy with total volume that can't be fully executed due to cap
        total_vol = 1500  # want to buy 1500 shares over 2 days
        max_pct = 0.5     # 50% of daily volume max, so max 500 shares per day (50% of 1000)
        # With 2 days and equal target, target_daily = 750, but cap is 500 each day, so we expect 500 each day => 1000 total executed, 500 remaining not executed.
        result = simulate_twap_strategy(self.test_data, total_vol, max_pct)
        total_executed = result["BuyVolume"].sum()
        self.assertEqual(total_executed, 1000)
        # Ensure no day exceeds 50% of volume
        for pct in result["PctOfVolume"]:
            self.assertLessEqual(pct, 50.0)
    
    def test_volume_participation_strategy(self):
        # Test volume participation strategy with same scenario
        total_vol = 1500
        max_pct = 0.5  # 50%
        result = simulate_volume_participation_strategy(self.test_data, total_vol, max_pct)
        total_executed = result["BuyVolume"].sum()
        # Volume participation will also cap at 500 per day, over 2 days = 1000
        self.assertEqual(total_executed, 1000)
        # If we reduce total_vol to 800, it should finish in 2 days with 500 day1 and 300 day2
        result2 = simulate_volume_participation_strategy(self.test_data, 800, max_pct)
        executed_volumes = result2["BuyVolume"].tolist()
        self.assertEqual(executed_volumes, [500, 300])

if __name__ == "__main__":
    unittest.main()
