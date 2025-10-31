"""
Ornstein-Uhlenbeck (OU) process.

The OU process is a mean-reverting stochastic process often used to model
interest rates, volatility, and other mean-reverting phenomena.
"""

import numpy as np
from typing import Tuple, Optional
from .base import StochasticProcess


class OrnsteinUhlenbeck(StochasticProcess):
    """
    Ornstein-Uhlenbeck (OU) mean-reverting process.

    The process satisfies the stochastic differential equation:
        dX(t) = θ(μ - X(t))dt + σ dW(t)

    where:
        - θ > 0 is the speed of mean reversion
        - μ is the long-term mean level
        - σ > 0 is the volatility
        - W(t) is a standard Brownian motion

    The solution is:
        X(t) = X(0)e^(-θt) + μ(1 - e^(-θt)) + σ∫₀ᵗ e^(-θ(t-s)) dW(s)

    Attributes
    ----------
    initial_value : float
        Initial value X(0)
    theta : float
        Speed of mean reversion
    mu : float
        Long-term mean level
    sigma : float
        Volatility coefficient
    random_seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        initial_value: float,
        theta: float,
        mu: float,
        sigma: float,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Ornstein-Uhlenbeck process.

        Parameters
        ----------
        initial_value : float
            Initial value X(0)
        theta : float
            Speed of mean reversion (must be positive)
        mu : float
            Long-term mean level
        sigma : float
            Volatility coefficient (must be positive)
        random_seed : int, optional
            Random seed for reproducibility
        """
        super().__init__(initial_value, random_seed)
        if theta <= 0:
            raise ValueError("theta must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
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
                self.theta * (self.mu - paths[:, i]) * dt + \
                self.sigma * dW[:, i]

        return t, paths

    def simulate_exact(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using the exact conditional distribution.

        For the OU process, we know that:
        X(t+dt) | X(t) ~ N(X(t)e^(-θdt) + μ(1-e^(-θdt)), σ²(1-e^(-2θdt))/(2θ))

        This is more accurate than Euler-Maruyama, especially for larger time steps.

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

        # Precompute constants
        exp_theta_dt = np.exp(-self.theta * dt)
        mean_factor = 1 - exp_theta_dt
        std_dev = self.sigma * np.sqrt((1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta))

        # Generate paths using exact transition
        for i in range(n_steps):
            conditional_mean = paths[:, i] * exp_theta_dt + self.mu * mean_factor
            paths[:, i + 1] = conditional_mean + std_dev * np.random.randn(n_paths)

        return t, paths

    def expected_value(self, t: float) -> float:
        """
        Compute the theoretical expected value E[X(t)].

        For OU process: E[X(t)] = X(0)e^(-θt) + μ(1 - e^(-θt))

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        float
            Expected value at time t
        """
        return self.initial_value * np.exp(-self.theta * t) + \
            self.mu * (1 - np.exp(-self.theta * t))

    def variance(self, t: float) -> float:
        """
        Compute the theoretical variance Var[X(t)].

        For OU process: Var[X(t)] = σ²(1 - e^(-2θt))/(2θ)

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        float
            Variance at time t
        """
        return self.sigma**2 * (1 - np.exp(-2 * self.theta * t)) / (2 * self.theta)

    def stationary_distribution_params(self) -> Tuple[float, float]:
        """
        Get the parameters of the stationary distribution.

        As t → ∞, the OU process converges to a normal distribution:
        X(∞) ~ N(μ, σ²/(2θ))

        Returns
        -------
        mean : float
            Stationary mean (= μ)
        variance : float
            Stationary variance (= σ²/(2θ))
        """
        return self.mu, self.sigma**2 / (2 * self.theta)

    def half_life(self) -> float:
        """
        Compute the half-life of mean reversion.

        The half-life is the time it takes for the process to move
        halfway towards the mean from any starting point.

        Returns
        -------
        float
            Half-life = ln(2)/θ
        """
        return np.log(2) / self.theta


# Alias for convenience
OU = OrnsteinUhlenbeck
