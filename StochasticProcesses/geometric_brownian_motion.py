"""
Geometric Brownian Motion (GBM) process.

The GBM is a continuous-time stochastic process where the logarithm of the
randomly varying quantity follows a Brownian motion with drift.
"""

import numpy as np
from typing import Tuple, Optional
from .base import StochasticProcess


class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion process.

    The process satisfies the stochastic differential equation:
        dS(t) = μ S(t) dt + σ S(t) dW(t)

    where:
        - μ is the drift coefficient (expected return)
        - σ is the volatility coefficient
        - W(t) is a standard Brownian motion

    The solution is:
        S(t) = S(0) * exp((μ - σ²/2)t + σ W(t))

    Attributes
    ----------
    initial_value : float
        Initial value S(0)
    mu : float
        Drift coefficient (expected return)
    sigma : float
        Volatility coefficient
    random_seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        initial_value: float,
        mu: float,
        sigma: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Geometric Brownian Motion process.

        Parameters
        ----------
        initial_value : float
            Initial value S(0)
        mu : float
            Drift coefficient (expected return)
        sigma : float
            Volatility coefficient
        random_seed : int, optional
            Random seed for reproducibility
        """
        super().__init__(initial_value, random_seed)
        self.mu = mu
        self.sigma = sigma

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using the exact solution of the GBM SDE.

        Parameters
        ----------
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int, optional
            Number of paths to simulate (default: 1)

        Returns
        -------
        t : np.ndarray
            Time grid of shape (n_steps + 1,)
        paths : np.ndarray
            Simulated paths of shape (n_paths, n_steps + 1)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        # Generate Brownian motion increments
        dW = np.sqrt(dt) * np.random.randn(n_paths, n_steps)

        # Compute cumulative sum to get W(t)
        W = np.zeros((n_paths, n_steps + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)

        # Compute S(t) using exact solution
        paths = self.initial_value * np.exp(
            (self.mu - 0.5 * self.sigma**2) * t + self.sigma * W
        )

        return t, paths

    def simulate_euler(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using the Euler-Maruyama discretization.

        This is an alternative to the exact solution, useful for comparison
        or when modifications to the standard GBM are needed.

        Parameters
        ----------
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int, optional
            Number of paths to simulate (default: 1)

        Returns
        -------
        t : np.ndarray
            Time grid of shape (n_steps + 1,)
        paths : np.ndarray
            Simulated paths of shape (n_paths, n_steps + 1)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.initial_value

        # Generate Brownian increments
        dW = np.sqrt(dt) * np.random.randn(n_paths, n_steps)

        # Euler-Maruyama scheme
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] + \
                self.mu * paths[:, i] * dt + \
                self.sigma * paths[:, i] * dW[:, i]

        return t, paths

    def expected_value(self, t: float) -> float:
        """
        Compute the theoretical expected value E[S(t)].

        For GBM: E[S(t)] = S(0) * exp(μt)

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        float
            Expected value at time t
        """
        return self.initial_value * np.exp(self.mu * t)

    def variance(self, t: float) -> float:
        """
        Compute the theoretical variance Var[S(t)].

        For GBM: Var[S(t)] = S(0)² * exp(2μt) * (exp(σ²t) - 1)

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        float
            Variance at time t
        """
        return self.initial_value**2 * np.exp(2 * self.mu * t) * \
            (np.exp(self.sigma**2 * t) - 1)

    def price_option(
        self,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        n_paths: int = 10000
    ) -> dict:
        """
        Price European options using Monte Carlo simulation.

        Parameters
        ----------
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free interest rate
        option_type : str
            'call' or 'put'
        n_paths : int
            Number of simulation paths

        Returns
        -------
        dict
            Dictionary containing price and standard error
        """
        # Simulate paths to maturity
        _, paths = self.simulate(T, n_steps=1, n_paths=n_paths)
        S_T = paths[:, -1]

        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # Discount to present value
        discount_factor = np.exp(-r * T)
        price = discount_factor * np.mean(payoffs)
        std_error = discount_factor * np.std(payoffs) / np.sqrt(n_paths)

        return {
            'price': price,
            'std_error': std_error
        }


# Alias for convenience
GBM = GeometricBrownianMotion
