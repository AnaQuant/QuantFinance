"""
Cox-Ingersoll-Ross (CIR) process.

The CIR process is a mean-reverting stochastic process commonly used to model
interest rates. Unlike the Ornstein-Uhlenbeck process, it guarantees
non-negative values under certain parameter conditions.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import ncx2
from .base import StochasticProcess


class CIRProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross (CIR) mean-reverting process.

    The process satisfies the stochastic differential equation:
        dX(t) = θ(μ - X(t))dt + σ√X(t) dW(t)

    where:
        - θ > 0 is the speed of mean reversion
        - μ > 0 is the long-term mean level
        - σ > 0 is the volatility
        - W(t) is a standard Brownian motion

    The solution is:
        X(t) = X(0)e^(-θt) + μ(1 - e^(-θt)) + σ∫₀ᵗ e^(-θ(t-s))√X(s) dW(s)

    The Feller condition (2θμ ≥ σ²) ensures the process stays strictly positive.

    Attributes
    ----------
    initial_value : float
        Initial value X(0) (must be positive)
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
        Initialize the CIR process.

        Parameters
        ----------
        initial_value : float
            Initial value X(0) (must be positive)
        theta : float
            Speed of mean reversion (must be positive)
        mu : float
            Long-term mean level (must be positive)
        sigma : float
            Volatility coefficient (must be positive)
        random_seed : int, optional
            Random seed for reproducibility
        """
        if initial_value <= 0:
            raise ValueError("initial_value must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if mu <= 0:
            raise ValueError("mu must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        super().__init__(initial_value, random_seed)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def satisfies_feller_condition(self) -> bool:
        """
        Check if the Feller condition is satisfied.

        The Feller condition (2θμ ≥ σ²) ensures the process remains
        strictly positive.

        Returns
        -------
        bool
            True if Feller condition is satisfied
        """
        return 2 * self.theta * self.mu >= self.sigma**2

    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1,
        method: str = 'euler'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths of the CIR process.

        Parameters
        ----------
        T : float
            Time horizon
        n_steps : int
            Number of time steps
        n_paths : int, optional
            Number of paths to simulate (default: 1)
        method : str, optional
            Simulation method: 'euler' or 'exact' (default: 'euler')

        Returns
        -------
        t : np.ndarray
            Time grid of shape (n_steps + 1,)
        paths : np.ndarray
            Simulated paths of shape (n_paths, n_steps + 1)
        """
        if method == 'euler':
            return self.simulate_euler(T, n_steps, n_paths)
        elif method == 'exact':
            return self.simulate_exact(T, n_steps, n_paths)
        else:
            raise ValueError("method must be 'euler' or 'exact'")

    def simulate_euler(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using the Euler-Maruyama discretization.

        Uses max(X, 0) to ensure non-negativity even if Feller condition
        is not satisfied.

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

        # Euler-Maruyama scheme with reflection at zero
        for i in range(n_steps):
            sqrt_X = np.sqrt(np.maximum(paths[:, i], 0))
            paths[:, i + 1] = paths[:, i] + \
                self.theta * (self.mu - paths[:, i]) * dt + \
                self.sigma * sqrt_X * dW[:, i]
            # Ensure non-negativity
            paths[:, i + 1] = np.maximum(paths[:, i + 1], 0)

        return t, paths

    def simulate_exact(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using the exact non-central chi-squared distribution.

        The conditional distribution X(t+dt) | X(t) follows a scaled
        non-central chi-squared distribution.

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

        # Parameters for the non-central chi-squared distribution
        exp_theta_dt = np.exp(-self.theta * dt)
        c = self.sigma**2 * (1 - exp_theta_dt) / (4 * self.theta)
        delta = 4 * self.theta * self.mu / self.sigma**2  # degrees of freedom

        # Generate paths
        for i in range(n_steps):
            # Non-centrality parameter
            lambda_param = 4 * self.theta * paths[:, i] * exp_theta_dt / \
                          (self.sigma**2 * (1 - exp_theta_dt))

            # Sample from scaled non-central chi-squared
            for j in range(n_paths):
                paths[j, i + 1] = c * ncx2.rvs(df=delta, nc=lambda_param[j])

        return t, paths

    def expected_value(self, t: float) -> float:
        """
        Compute the theoretical expected value E[X(t)].

        For CIR process: E[X(t)] = X(0)e^(-θt) + μ(1 - e^(-θt))

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

        For CIR process:
        Var[X(t)] = (σ²/θ)X(0)(e^(-θt) - e^(-2θt)) + (μσ²/(2θ))(1 - e^(-θt))²

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        float
            Variance at time t
        """
        exp_theta_t = np.exp(-self.theta * t)
        exp_2theta_t = np.exp(-2 * self.theta * t)

        term1 = (self.sigma**2 / self.theta) * self.initial_value * \
                (exp_theta_t - exp_2theta_t)
        term2 = (self.mu * self.sigma**2 / (2 * self.theta)) * \
                (1 - exp_theta_t)**2

        return term1 + term2

    def stationary_distribution_params(self) -> Tuple[float, float]:
        """
        Get the parameters of the stationary distribution.

        As t → ∞, the CIR process converges to a Gamma distribution
        with mean μ and variance σ²μ/(2θ).

        Returns
        -------
        mean : float
            Stationary mean (= μ)
        variance : float
            Stationary variance (= σ²μ/(2θ))
        """
        return self.mu, self.sigma**2 * self.mu / (2 * self.theta)

    def marginal_pdf_params(self, t: float) -> dict:
        """
        Get the parameters for the marginal distribution at time t.

        The marginal distribution X(t) | X(0) follows a scaled non-central
        chi-squared distribution.

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        dict
            Dictionary with 'scale', 'df' (degrees of freedom), and 'nc' (non-centrality)
        """
        exp_theta_t = np.exp(-self.theta * t)
        c = self.sigma**2 * (1 - exp_theta_t) / (4 * self.theta)
        delta = 4 * self.theta * self.mu / self.sigma**2
        lambda_param = 4 * self.theta * self.initial_value * exp_theta_t / \
                      (self.sigma**2 * (1 - exp_theta_t))

        return {
            'scale': c,
            'df': delta,
            'nc': lambda_param
        }
