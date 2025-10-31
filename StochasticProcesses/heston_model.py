"""
Heston Stochastic Volatility Model.

The Heston model is a stochastic volatility model where both the asset price
and its volatility follow stochastic processes.
"""

import numpy as np
from typing import Tuple, Optional
from .base import StochasticProcess


class HestonModel(StochasticProcess):
    """
    Heston stochastic volatility model.

    The model consists of two coupled stochastic differential equations:
        dS(t) = μ S(t) dt + √v(t) S(t) dW₁(t)
        dv(t) = κ(θ - v(t))dt + σᵥ√v(t) dW₂(t)

    where:
        - S(t) is the asset price
        - v(t) is the variance (volatility squared)
        - μ is the drift of the asset
        - κ is the speed of mean reversion for variance
        - θ is the long-term variance level
        - σᵥ is the volatility of variance
        - W₁(t) and W₂(t) are correlated Brownian motions with correlation ρ

    The Feller condition (2κθ ≥ σᵥ²) ensures variance stays positive.

    Attributes
    ----------
    initial_value : float
        Initial asset price S(0)
    initial_variance : float
        Initial variance v(0)
    mu : float
        Drift of asset price
    kappa : float
        Speed of mean reversion for variance
    theta : float
        Long-term variance level
    sigma_v : float
        Volatility of variance
    rho : float
        Correlation between asset and variance Brownian motions
    random_seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        initial_value: float,
        initial_variance: float,
        mu: float,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Heston model.

        Parameters
        ----------
        initial_value : float
            Initial asset price S(0) (must be positive)
        initial_variance : float
            Initial variance v(0) (must be positive)
        mu : float
            Drift of asset price
        kappa : float
            Speed of mean reversion for variance (must be positive)
        theta : float
            Long-term variance level (must be positive)
        sigma_v : float
            Volatility of variance (must be positive)
        rho : float
            Correlation between asset and variance (-1 ≤ ρ ≤ 1)
        random_seed : int, optional
            Random seed for reproducibility
        """
        if initial_value <= 0:
            raise ValueError("initial_value must be positive")
        if initial_variance <= 0:
            raise ValueError("initial_variance must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if sigma_v <= 0:
            raise ValueError("sigma_v must be positive")
        if not -1 <= rho <= 1:
            raise ValueError("rho must be between -1 and 1")

        super().__init__(initial_value, random_seed)
        self.initial_variance = initial_variance
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def satisfies_feller_condition(self) -> bool:
        """
        Check if the Feller condition is satisfied for the variance process.

        The Feller condition (2κθ ≥ σᵥ²) ensures the variance process
        remains strictly positive.

        Returns
        -------
        bool
            True if Feller condition is satisfied
        """
        return 2 * self.kappa * self.theta >= self.sigma_v**2

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths using the Euler-Maruyama discretization.

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
        price_paths : np.ndarray
            Simulated asset price paths of shape (n_paths, n_steps + 1)
        variance_paths : np.ndarray
            Simulated variance paths of shape (n_paths, n_steps + 1)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        # Initialize paths
        price_paths = np.zeros((n_paths, n_steps + 1))
        variance_paths = np.zeros((n_paths, n_steps + 1))
        price_paths[:, 0] = self.initial_value
        variance_paths[:, 0] = self.initial_variance

        # Generate correlated Brownian increments
        # W1 and W2 with correlation rho
        dW1 = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
        dW2_indep = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * dW2_indep

        # Euler-Maruyama scheme
        for i in range(n_steps):
            # Ensure variance stays positive (full truncation scheme)
            v_current = np.maximum(variance_paths[:, i], 0)
            sqrt_v = np.sqrt(v_current)

            # Update asset price
            price_paths[:, i + 1] = price_paths[:, i] + \
                self.mu * price_paths[:, i] * dt + \
                sqrt_v * price_paths[:, i] * dW1[:, i]

            # Update variance (CIR process for variance)
            variance_paths[:, i + 1] = variance_paths[:, i] + \
                self.kappa * (self.theta - v_current) * dt + \
                self.sigma_v * sqrt_v * dW2[:, i]

            # Ensure variance stays non-negative
            variance_paths[:, i + 1] = np.maximum(variance_paths[:, i + 1], 0)

        return t, price_paths, variance_paths

    def simulate_milstein(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate paths using the Milstein discretization for better accuracy.

        The Milstein scheme includes second-order terms for improved accuracy.

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
        price_paths : np.ndarray
            Simulated asset price paths of shape (n_paths, n_steps + 1)
        variance_paths : np.ndarray
            Simulated variance paths of shape (n_paths, n_steps + 1)
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        # Initialize paths
        price_paths = np.zeros((n_paths, n_steps + 1))
        variance_paths = np.zeros((n_paths, n_steps + 1))
        price_paths[:, 0] = self.initial_value
        variance_paths[:, 0] = self.initial_variance

        # Generate correlated Brownian increments
        dW1 = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
        dW2_indep = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * dW2_indep

        # Milstein scheme
        for i in range(n_steps):
            v_current = np.maximum(variance_paths[:, i], 0)
            sqrt_v = np.sqrt(v_current)

            # Update asset price (Milstein for GBM part)
            price_paths[:, i + 1] = price_paths[:, i] + \
                self.mu * price_paths[:, i] * dt + \
                sqrt_v * price_paths[:, i] * dW1[:, i] + \
                0.5 * v_current * price_paths[:, i] * (dW1[:, i]**2 - dt)

            # Update variance (Milstein for CIR part)
            variance_paths[:, i + 1] = variance_paths[:, i] + \
                self.kappa * (self.theta - v_current) * dt + \
                self.sigma_v * sqrt_v * dW2[:, i] + \
                0.25 * self.sigma_v**2 * (dW2[:, i]**2 - dt)

            # Ensure variance stays non-negative
            variance_paths[:, i + 1] = np.maximum(variance_paths[:, i + 1], 0)

        return t, price_paths, variance_paths

    def get_volatility_paths(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate and return volatility (not variance) paths.

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
        volatility_paths : np.ndarray
            Simulated volatility paths of shape (n_paths, n_steps + 1)
        """
        t, _, variance_paths = self.simulate(T, n_steps, n_paths)
        volatility_paths = np.sqrt(variance_paths)
        return t, volatility_paths
