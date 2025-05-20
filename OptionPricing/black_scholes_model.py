import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


class BlackScholesModel:
    """
    Black-Scholes option pricing model, based on Geometric Brownian Motion.
    """

    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float):
        """
        Initialize the Black-Scholes model.

        Parameters:
        -----------
        S0: float
            Current stock price
        K: float
            Strike price
        r: float
            Risk-free interest rate (annualized)
        sigma: float
            Volatility (annualized)
        T: float
            Time to maturity (in years)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def _calculate_d1(self) -> float:
        """Calculate the d1 parameter for Black-Scholes formula."""
        return (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _calculate_d2(self) -> float:
        """Calculate the d2 parameter for Black-Scholes formula."""
        return self._calculate_d1() - self.sigma * np.sqrt(self.T)

    def price_call_option(self) -> float:
        """
        Calculate the price of a European call option.

        Returns:
        --------
        float
            Call option price
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()

        return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

    def price_put_option(self) -> float:
        """
        Calculate the price of a European put option.

        Returns:
        --------
        float
            Put option price
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()

        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)

    def calculate_greeks(self) -> dict:
        """
        Calculate option Greeks (delta, gamma, theta, vega, rho).

        Returns:
        --------
        dict
            Dictionary containing the option Greeks
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()

        # Delta
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1

        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(self.T))

        # Theta (time decay)
        theta_call = -self.S0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) - \
                     self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        theta_put = -self.S0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) + \
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)

        # Vega (sensitivity to volatility, same for call and put)
        vega = self.S0 * np.sqrt(self.T) * norm.pdf(d1)

        # Rho (sensitivity to interest rate)
        rho_call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        rho_put = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        return {
            "call_delta": call_delta,
            "put_delta": put_delta,
            "gamma": gamma,
            "call_theta": theta_call,
            "put_theta": theta_put,
            "vega": vega,
            "call_rho": rho_call,
            "put_rho": rho_put
        }

    def plot_option_value_vs_underlying(self, price_range_factor: float = 0.5):
        """
        Plot option values across a range of underlying prices.

        Parameters:
        -----------
        price_range_factor: float
            Factor to determine the price range around current price
        """
        price_range = np.linspace(self.S0 * (1 - price_range_factor),
                                  self.S0 * (1 + price_range_factor),
                                  100)

        call_values = []
        put_values = []

        for price in price_range:
            temp_model = BlackScholesModel(price, self.K, self.r, self.sigma, self.T)
            call_values.append(temp_model.price_call_option())
            put_values.append(temp_model.price_put_option())

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(price_range, call_values)
        plt.axvline(x=self.K, color='r', linestyle='--', label=f'Strike: {self.K}')
        plt.title('Call Option Value vs Underlying Price')
        plt.xlabel('Underlying Price')
        plt.ylabel('Option Value')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(price_range, put_values)
        plt.axvline(x=self.K, color='r', linestyle='--', label=f'Strike: {self.K}')
        plt.title('Put Option Value vs Underlying Price')
        plt.xlabel('Underlying Price')
        plt.ylabel('Option Value')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def monte_carlo_option_pricing(self, num_paths: int = 10000, random_seed: int = None):
        """
        Price options using Monte Carlo simulation with GBM.

        Parameters:
        -----------
        num_paths: int
            Number of price paths to simulate
        random_seed: int, optional
            Random seed for reproducibility

        Returns:
        --------
        dict
            Dictionary containing option prices and standard errors
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random normally distributed numbers for simulation
        z = np.random.normal(0, 1, num_paths)

        # Calculate the final price using the GBM solution
        ST = self.S0 * np.exp((self.r - 0.5 * self.sigma ** 2) * self.T + self.sigma * np.sqrt(self.T) * z)

        # Calculate option payoffs at maturity
        call_payoffs = np.maximum(ST - self.K, 0)
        put_payoffs = np.maximum(self.K - ST, 0)

        # Discount the payoffs to present value
        discount_factor = np.exp(-self.r * self.T)
        call_prices = discount_factor * call_payoffs
        put_prices = discount_factor * put_payoffs

        # Calculate average prices and standard errors
        call_price = np.mean(call_prices)
        put_price = np.mean(put_prices)
        call_se = np.std(call_prices) / np.sqrt(num_paths)
        put_se = np.std(put_prices) / np.sqrt(num_paths)

        return {
            "call_price": call_price,
            "put_price": put_price,
            "call_std_error": call_se,
            "put_std_error": put_se,
            "analytical_call": self.price_call_option(),
            "analytical_put": self.price_put_option()
        }


# Example usage
if __name__ == "__main__":
    # Parameters
    S0 = 100.0  # Current stock price
    K = 100.0  # Strike price (at-the-money)
    r = 0.05  # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    T = 1.0  # Time to maturity (1 year)

    # Create Black-Scholes model
    bs_model = BlackScholesModel(S0=S0, K=K, r=r, sigma=sigma, T=T)

    # Price options
    call_price = bs_model.price_call_option()
    put_price = bs_model.price_put_option()

    print(f"Black-Scholes Option Prices:")
    print(f"Call Option: ${call_price:.4f}")
    print(f"Put Option: ${put_price:.4f}")

    # Calculate Greeks
    greeks = bs_model.calculate_greeks()
    print("\nOption Greeks:")
    print(f"Call Delta: {greeks['call_delta']:.4f}")
    print(f"Put Delta: {greeks['put_delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Call Theta: {greeks['call_theta']:.4f}")
    print(f"Put Theta: {greeks['put_theta']:.4f}")

    # Plot option values vs. underlying price
    bs_model.plot_option_value_vs_underlying()

    # Monte Carlo simulation
    mc_results = bs_model.monte_carlo_option_pricing(num_paths=100000, random_seed=42)
    print("\nMonte Carlo Option Pricing:")
    print(f"MC Call Price: ${mc_results['call_price']:.4f} (SE: {mc_results['call_std_error']:.6f})")
    print(f"MC Put Price: ${mc_results['put_price']:.4f} (SE: {mc_results['put_std_error']:.6f})")
    print(f"Analytical Call: ${mc_results['analytical_call']:.4f}")
    print(f"Analytical Put: ${mc_results['analytical_put']:.4f}")
