import pandas as pd
import numpy as np


class Portfolio:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self._validate_columns(["quantity", "price", "sector", "beta", "market_cap", "currency"])

        # Calculate market value per position
        self.data['Market Value'] = self.data['quantity'] * self.data['price']
        self.total_value = self.data['Market Value'].sum()

    def _validate_columns(self, required_columns):
        """Ensure the required columns exist in the dataset."""
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def get_portfolio(self):
        return self.data


class PortfolioExposures:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def sector_exposure(self):
        """Calculates sector exposure as a percentage of total portfolio value."""
        return self.portfolio.data.groupby('sector')['Market Value'].sum() / self.portfolio.total_value

    def beta_exposure(self):
        """Calculates the weighted average beta of the portfolio."""
        return (self.portfolio.data['beta'] * self.portfolio.data['Market Value']).sum() / self.portfolio.total_value

    def market_cap_exposure(self):
        """Calculates market cap exposure as a percentage of total portfolio value."""
        return self.portfolio.data.groupby('market_cap')['Market Value'].sum() / self.portfolio.total_value

    def historical_var(self, returns, confidence=0.95):
        """Calculates historical Value at Risk (VaR)."""
        return np.percentile(returns, (1 - confidence) * 100)

    def volatility_exposure(self):
        """Calculates weighted average volatility exposure."""
        if 'Volatility' in self.portfolio.data.columns:
            return (self.portfolio.data['Volatility'] * self.portfolio.data[
                'Market Value']).sum() / self.portfolio.total_value
        else:
            return "Volatility data not available"


if __name__ == "__main__":
    portfolio = Portfolio("portfolio_exposures.csv")
    exposures = PortfolioExposures(portfolio)

    print("Sector Exposure:\n", exposures.sector_exposure())
    print("Portfolio Beta:", exposures.beta_exposure())
    print("Market Cap Exposure:\n", exposures.market_cap_exposure())

    # Example usage for VaR calculation (assumes past returns are available)
    # In real cases, you would need a time series of portfolio returns
    # Example dummy returns: replace with real data
    sample_returns = np.random.normal(0, 0.02, 1000)  # Simulated daily returns
    print("Portfolio Historical VaR (95% confidence):", exposures.historical_var(sample_returns))

    # Optional volatility exposure if 'Volatility' column exists
    print("Volatility Exposure:", exposures.volatility_exposure())