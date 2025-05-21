import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import math


# Black-Scholes Option Pricing Functions
def d1(S, K, r, sigma, T):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, r, sigma, T):
    return d1(S, K, r, sigma, T) - sigma * np.sqrt(T)


def bs_call_price(S, K, r, sigma, T):
    """Calculate Black-Scholes price for a call option"""
    if T <= 0:
        return max(0, S - K)
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def bs_put_price(S, K, r, sigma, T):
    """Calculate Black-Scholes price for a put option"""
    if T <= 0:
        return max(0, K - S)
    D1 = d1(S, K, r, sigma, T)
    D2 = d2(S, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)


# Option Greeks
def bs_call_delta(S, K, r, sigma, T):
    """Calculate delta for a call option"""
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(d1(S, K, r, sigma, T))


def bs_put_delta(S, K, r, sigma, T):
    """Calculate delta for a put option"""
    if T <= 0:
        return 0.0 if S > K else -1.0
    return -norm.cdf(-d1(S, K, r, sigma, T))


def bs_gamma(S, K, r, sigma, T):
    """Calculate gamma for an option (same for calls and puts)"""
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, r, sigma, T):
    """Calculate vega for an option (same for calls and puts)"""
    if T <= 0:
        return 0.0
    D1 = d1(S, K, r, sigma, T)
    return S * np.sqrt(T) * norm.pdf(D1) / 100  # Divided by 100 to get sensitivity to 1% change in vol


class Portfolio:
    def __init__(self):
        self.positions = []

    def add_stock(self, symbol, quantity, price, volatility):
        """Add a stock position to the portfolio"""
        self.positions.append({
            'type': 'stock',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'volatility': volatility
        })

    def add_option(self, symbol, option_type, quantity, underlying_price, strike,
                   risk_free_rate, volatility, time_to_expiry):
        """Add an option position to the portfolio"""
        self.positions.append({
            'type': 'option',
            'option_type': option_type,  # 'call' or 'put'
            'symbol': symbol,
            'quantity': quantity,
            'underlying_price': underlying_price,
            'strike': strike,
            'risk_free_rate': risk_free_rate,
            'volatility': volatility,
            'time_to_expiry': time_to_expiry
        })

    def calculate_portfolio_value(self):
        """Calculate the total portfolio value"""
        total_value = 0

        for position in self.positions:
            if position['type'] == 'stock':
                total_value += position['quantity'] * position['price']
            elif position['type'] == 'option':
                if position['option_type'] == 'call':
                    option_price = bs_call_price(
                        position['underlying_price'],
                        position['strike'],
                        position['risk_free_rate'],
                        position['volatility'],
                        position['time_to_expiry']
                    )
                else:  # put option
                    option_price = bs_put_price(
                        position['underlying_price'],
                        position['strike'],
                        position['risk_free_rate'],
                        position['volatility'],
                        position['time_to_expiry']
                    )
                total_value += position['quantity'] * option_price * 100  # Assuming 1 option = 100 shares

        return total_value

    def calculate_delta_normal_var(self, confidence_level=0.95, time_horizon=1):
        """
        Calculate Delta-Normal VaR for the portfolio

        Parameters:
        confidence_level (float): Confidence level for VaR (e.g., 0.95 for 95%)
        time_horizon (int): Time horizon in days

        Returns:
        float: VaR value
        """
        # Group positions by underlying
        underlyings = {}

        for position in self.positions:
            if position['type'] == 'stock':
                symbol = position['symbol']
                if symbol not in underlyings:
                    underlyings[symbol] = {
                        'price': position['price'],
                        'volatility': position['volatility'],
                        'delta_exposure': 0
                    }
                # Add stock delta exposure (always 1.0 for a stock)
                underlyings[symbol]['delta_exposure'] += position['quantity']

            elif position['type'] == 'option':
                symbol = position['symbol']
                if symbol not in underlyings:
                    underlyings[symbol] = {
                        'price': position['underlying_price'],
                        'volatility': position['volatility'],
                        'delta_exposure': 0
                    }

                # Calculate option delta
                if position['option_type'] == 'call':
                    delta = bs_call_delta(
                        position['underlying_price'],
                        position['strike'],
                        position['risk_free_rate'],
                        position['volatility'],
                        position['time_to_expiry']
                    )
                else:  # put option
                    delta = bs_put_delta(
                        position['underlying_price'],
                        position['strike'],
                        position['risk_free_rate'],
                        position['volatility'],
                        position['time_to_expiry']
                    )

                # Add option delta exposure (multiplied by 100 as each option controls 100 shares)
                underlyings[symbol]['delta_exposure'] += position['quantity'] * delta * 100

        # Calculate portfolio variance
        portfolio_variance = 0

        # For simplicity, we assume no correlation between different underlyings
        # In a real implementation, you would use a correlation matrix
        for symbol, data in underlyings.items():
            price = data['price']
            volatility = data['volatility']
            delta_exposure = data['delta_exposure']

            # Contribution to portfolio variance
            position_variance = (delta_exposure * price * volatility) ** 2
            portfolio_variance += position_variance

        # Calculate VaR
        portfolio_stddev = np.sqrt(portfolio_variance) * np.sqrt(
            time_horizon / 252)  # Scale by time (assuming 252 trading days)
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -z_score * portfolio_stddev

        return var


# Example usage
def run_example():
    # Create a portfolio
    portfolio = Portfolio()

    # Add positions
    # Stock: 1000 shares of AAPL at $200 with 30% annualized volatility
    portfolio.add_stock('AAPL', 1000, 200, 0.30)

    # Call option: 5 AAPL call contracts, strike $210, expiry in 0.25 years (3 months)
    portfolio.add_option('AAPL', 'call', 5, 200, 210, 0.05, 0.30, 0.25)

    # Put option: 3 AAPL put contracts, strike $190, expiry in 0.25 years (3 months)
    portfolio.add_option('AAPL', 'put', 3, 200, 190, 0.05, 0.30, 0.25)

    # Calculate portfolio value
    portfolio_value = portfolio.calculate_portfolio_value()
    print(f"Portfolio Value: ${portfolio_value:,.2f}")

    # Calculate 1-day VaR at 95% confidence level
    var_95 = portfolio.calculate_delta_normal_var(confidence_level=0.95, time_horizon=1)
    print(f"1-Day 95% Delta-Normal VaR: ${var_95:,.2f}")

    # Calculate 1-day VaR at 99% confidence level
    var_99 = portfolio.calculate_delta_normal_var(confidence_level=0.99, time_horizon=1)
    print(f"1-Day 99% Delta-Normal VaR: ${var_99:,.2f}")

    # Calculate 10-day VaR at 99% confidence level (for regulatory purposes)
    var_99_10day = portfolio.calculate_delta_normal_var(confidence_level=0.99, time_horizon=10)
    print(f"10-Day 99% Delta-Normal VaR: ${var_99_10day:,.2f}")

    # VaR as a percentage of portfolio value
    print(f"1-Day 95% VaR as % of Portfolio: {(var_95 / portfolio_value * 100):.2f}%")
    print(f"1-Day 99% VaR as % of Portfolio: {(var_99 / portfolio_value * 100):.2f}%")


if __name__ == "__main__":
    run_example()
