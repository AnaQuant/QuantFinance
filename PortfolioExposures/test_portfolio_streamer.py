from PortfolioExposures.portfolio_streamer import PortfolioManager
import unittest

class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.manager = PortfolioManager()
        self.manager.portfolios = {
            "TECH": {"AAPL": 100, "MSFT": 200},
            "INDUSTRIALS": {"TECH": 2, "AUTOS": 3},
            "AUTOS": {"TSLA": 50, "FORD": 100}
        }
        self.manager.stock_prices = {"AAPL": 150, "MSFT": 300, "TSLA": 700, "FORD": 50}

    def test_calculate_portfolio_value(self):
        self.assertEqual(self.manager.calculate_portfolio_value("TECH"), 100 * 150 + 200 * 300)
        self.assertEqual(self.manager.calculate_portfolio_value("AUTOS"), 50 * 700 + 100 * 50)
        self.assertEqual(self.manager.calculate_portfolio_value("INDUSTRIALS"),
                         2 * (100 * 150 + 200 * 300) + 3 * (50 * 700 + 100 * 50))

    def test_update_stock_price_affects_portfolio(self):
        self.manager.stock_prices["AAPL"] = 160
        self.manager.portfolio_cache.clear()
        self.assertEqual(self.manager.calculate_portfolio_value("TECH"), 100 * 160 + 200 * 300)

    def test_missing_price_returns_none(self):
        del self.manager.stock_prices["AAPL"]
        self.manager.portfolio_cache.clear()
        self.assertIsNone(self.manager.calculate_portfolio_value("TECH"))

