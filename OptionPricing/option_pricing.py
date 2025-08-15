"""
Option Pricing Framework
========================
A modular framework for pricing options, starting with Black-Scholes.
This framework is designed to be extended with additional models and features.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, List, Dict, Tuple, Callable


# ---- Enums and Data Types ----

class OptionType(Enum):
    """Enum for option types."""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Enum for option exercise styles."""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionParams:
    """Class for storing option parameters."""
    S: float  # Current stock price
    K: float  # Strike price
    T: float  # Time to maturity (in years)
    r: float  # Risk-free rate (annualized)
    sigma: float  # Volatility (annualized)
    q: float = 0.0  # Dividend yield (annualized)
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN


# ---- Base Classes ----

class PricingModel(ABC):
    """Abstract base class for option pricing models."""

    @abstractmethod
    def price(self, params: OptionParams) -> float:
        """Calculate the option price."""
        pass

    @abstractmethod
    def greeks(self, params: OptionParams) -> Dict[str, float]:
        """Calculate the option Greeks."""
        pass


# ---- Black-Scholes Model ----

class BlackScholes(PricingModel):
    """Black-Scholes option pricing model."""

    def _d1(self, params: OptionParams) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        return (np.log(params.S / params.K) +
                (params.r - params.q + 0.5 * params.sigma ** 2) * params.T) / \
            (params.sigma * np.sqrt(params.T))

    def _d2(self, params: OptionParams) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return self._d1(params) - params.sigma * np.sqrt(params.T)

    def price(self, params: OptionParams) -> float:
        """Calculate the option price using Black-Scholes formula."""
        if params.exercise_style != ExerciseStyle.EUROPEAN:
            raise ValueError("Black-Scholes model only supports European options")

        d1 = self._d1(params)
        d2 = self._d2(params)

        if params.option_type == OptionType.CALL:
            return params.S * np.exp(-params.q * params.T) * norm.cdf(d1) - \
                params.K * np.exp(-params.r * params.T) * norm.cdf(d2)
        else:  # PUT
            return params.K * np.exp(-params.r * params.T) * norm.cdf(-d2) - \
                params.S * np.exp(-params.q * params.T) * norm.cdf(-d1)

    def greeks(self, params: OptionParams) -> Dict[str, float]:
        """Calculate the option Greeks for Black-Scholes."""
        d1 = self._d1(params)
        d2 = self._d2(params)

        # Factors used in calculations
        S = params.S
        K = params.K
        T = params.T
        r = params.r
        q = params.q
        sigma = params.sigma
        sqrt_T = np.sqrt(T)
        exp_qt = np.exp(-q * T)
        exp_rt = np.exp(-r * T)

        # Normal distribution values
        N_d1 = norm.cdf(d1)
        N_minus_d1 = norm.cdf(-d1)
        N_d2 = norm.cdf(d2)
        N_minus_d2 = norm.cdf(-d2)
        n_d1 = norm.pdf(d1)

        greeks = {}

        # Delta
        if params.option_type == OptionType.CALL:
            greeks["delta"] = exp_qt * N_d1
        else:  # PUT
            greeks["delta"] = exp_qt * (N_d1 - 1)

        # Gamma (same for call and put)
        greeks["gamma"] = exp_qt * n_d1 / (S * sigma * sqrt_T)

        # Theta
        if params.option_type == OptionType.CALL:
            greeks["theta"] = -exp_qt * S * n_d1 * sigma / (2 * sqrt_T) - \
                              r * K * exp_rt * N_d2 + q * S * exp_qt * N_d1
        else:  # PUT
            greeks["theta"] = -exp_qt * S * n_d1 * sigma / (2 * sqrt_T) + \
                              r * K * exp_rt * N_minus_d2 - q * S * exp_qt * N_minus_d1

        # Vega (same for call and put)
        greeks["vega"] = S * exp_qt * n_d1 * sqrt_T

        # Rho
        if params.option_type == OptionType.CALL:
            greeks["rho"] = K * T * exp_rt * N_d2
        else:  # PUT
            greeks["rho"] = -K * T * exp_rt * N_minus_d2

        return greeks


# ---- Monte Carlo Simulation ----

class MonteCarlo(PricingModel):
    """Monte Carlo simulation for option pricing."""

    def __init__(self, n_paths: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator.

        Args:
            n_paths: Number of simulation paths
            random_seed: Seed for random number generator
        """
        self.n_paths = n_paths
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_paths(self, params: OptionParams, n_steps: int = 252) -> np.ndarray:
        """
        Simulate stock price paths using geometric Brownian motion.

        Args:
            params: Option parameters
            n_steps: Number of time steps in the simulation

        Returns:
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        dt = params.T / n_steps
        drift = (params.r - params.q - 0.5 * params.sigma ** 2) * dt
        diffusion = params.sigma * np.sqrt(dt)

        Z = np.random.normal(0, 1, size=(self.n_paths, n_steps))
        increments = drift + diffusion * Z

        # Initialize paths array with starting price
        paths = np.zeros((self.n_paths, n_steps + 1))
        paths[:, 0] = params.S

        # Compute paths using cumulative sum of increments
        for i in range(1, n_steps + 1):
            paths[:, i] = paths[:, i - 1] * np.exp(increments[:, i - 1])

        return paths

    def price(self, params: OptionParams, n_steps: int = 252) -> float:
        """
        Price an option using Monte Carlo simulation.

        Args:
            params: Option parameters
            n_steps: Number of time steps in the simulation

        Returns:
            Estimated option price
        """
        paths = self.simulate_paths(params, n_steps)

        # Extract final stock prices
        final_prices = paths[:, -1]

        # Calculate payoffs
        if params.option_type == OptionType.CALL:
            payoffs = np.maximum(final_prices - params.K, 0)
        else:  # PUT
            payoffs = np.maximum(params.K - final_prices, 0)

        # European option pricing
        if params.exercise_style == ExerciseStyle.EUROPEAN:
            # Discount payoffs and take mean
            price = np.exp(-params.r * params.T) * np.mean(payoffs)
            return price
        else:
            raise NotImplementedError("American option pricing not implemented yet in Monte Carlo")

    def greeks(self, params: OptionParams) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences with Monte Carlo simulation.

        Args:
            params: Option parameters

        Returns:
            Dictionary of Greeks
        """
        # Small perturbations for finite differences
        h_S = params.S * 0.01  # 1% of stock price
        h_sigma = 0.005  # 0.5% volatility
        h_r = 0.0025  # 0.25% interest rate
        h_T = 1 / 365  # 1 day

        # Base price
        base_price = self.price(params)

        # Delta
        params_up_S = OptionParams(**vars(params))
        params_up_S.S += h_S
        delta = (self.price(params_up_S) - base_price) / h_S

        # Gamma
        params_down_S = OptionParams(**vars(params))
        params_down_S.S -= h_S
        gamma = (self.price(params_up_S) - 2 * base_price + self.price(params_down_S)) / (h_S ** 2)

        # Vega
        params_up_sigma = OptionParams(**vars(params))
        params_up_sigma.sigma += h_sigma
        vega = (self.price(params_up_sigma) - base_price) / h_sigma

        # Theta
        params_up_T = OptionParams(**vars(params))
        params_up_T.T += h_T
        theta = (self.price(params_up_T) - base_price) / h_T

        # Rho
        params_up_r = OptionParams(**vars(params))
        params_up_r.r += h_r
        rho = (self.price(params_up_r) - base_price) / h_r

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }


# ---- Option Analyzer ----

class OptionAnalyzer:
    """
    Class for analyzing options with different pricing models.
    Provides utilities for parameter analysis and visualization.
    """

    def __init__(self, model: PricingModel):
        """
        Initialize the analyzer with a pricing model.

        Args:
            model: Pricing model to use
        """
        self.model = model

    def parameter_sweep(self,
                        base_params: OptionParams,
                        param_name: str,
                        param_range: List[float]) -> Tuple[List[float], List[float]]:
        """
        Analyze option price across a range of parameter values.

        Args:
            base_params: Base option parameters
            param_name: Name of parameter to vary ('S', 'K', 'T', 'r', 'sigma', 'q')
            param_range: Range of parameter values to analyze

        Returns:
            Tuple of (parameter values, option prices)
        """
        prices = []
        for value in param_range:
            # Create a copy of base parameters and update the specified parameter
            params = OptionParams(**vars(base_params))
            setattr(params, param_name, value)

            # Calculate price with updated parameters
            price = self.model.price(params)
            prices.append(price)

        return param_range, prices

    def plot_parameter_sensitivity(self,
                                   base_params: OptionParams,
                                   param_name: str,
                                   param_range: List[float],
                                   title: Optional[str] = None,
                                   ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot option price sensitivity to a parameter.

        Args:
            base_params: Base option parameters
            param_name: Name of parameter to vary
            param_range: Range of parameter values
            title: Plot title
            ax: Matplotlib axes for plotting

        Returns:
            Matplotlib axes with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        x_values, prices = self.parameter_sweep(base_params, param_name, param_range)

        ax.plot(x_values, prices, marker='o', linestyle='-')

        param_labels = {
            'S': 'Stock Price',
            'K': 'Strike Price',
            'T': 'Time to Maturity (years)',
            'r': 'Risk-free Rate',
            'sigma': 'Volatility',
            'q': 'Dividend Yield'
        }

        ax.set_xlabel(param_labels.get(param_name, param_name))
        ax.set_ylabel('Option Price')

        if title is None:
            option_type = 'Call' if base_params.option_type == OptionType.CALL else 'Put'
            title = f'{option_type} Option Price vs {param_labels.get(param_name, param_name)}'

        ax.set_title(title)
        ax.grid(True)

        return ax

    def plot_greeks(self,
                    base_params: OptionParams,
                    param_name: str,
                    param_range: List[float]) -> None:
        """
        Plot option Greeks sensitivity to a parameter.

        Args:
            base_params: Base option parameters
            param_name: Name of parameter to vary
            param_range: Range of parameter values
        """
        # Calculate Greeks for each parameter value
        greek_values = {
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': [],
            'rho': []
        }

        for value in param_range:
            params = OptionParams(**vars(base_params))
            setattr(params, param_name, value)

            greeks = self.model.greeks(params)
            for greek, val in greeks.items():
                greek_values[greek].append(val)

        # Create subplots for each Greek
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        axs = axs.flatten()

        param_labels = {
            'S': 'Stock Price',
            'K': 'Strike Price',
            'T': 'Time to Maturity (years)',
            'r': 'Risk-free Rate',
            'sigma': 'Volatility',
            'q': 'Dividend Yield'
        }

        option_type = 'Call' if base_params.option_type == OptionType.CALL else 'Put'

        # Plot each Greek
        for i, greek in enumerate(greek_values.keys()):
            if i < len(axs):  # Safety check
                axs[i].plot(param_range, greek_values[greek], marker='o', linestyle='-')
                axs[i].set_title(f'{greek.capitalize()} vs {param_labels.get(param_name, param_name)}')
                axs[i].set_xlabel(param_labels.get(param_name, param_name))
                axs[i].set_ylabel(greek.capitalize())
                axs[i].grid(True)

        # Remove unused subplot if any
        if len(greek_values) < len(axs):
            fig.delaxes(axs[-1])

        fig.suptitle(f'{option_type} Option Greeks Sensitivity to {param_labels.get(param_name, param_name)}',
                     fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        plt.show()


# ---- Factory for Models ----

class ModelFactory:
    """Factory class for creating pricing models."""

    @staticmethod
    def create_model(model_type: str, **kwargs) -> PricingModel:
        """
        Create a pricing model instance.

        Args:
            model_type: Type of model to create ('black_scholes', 'monte_carlo', etc.)
            **kwargs: Additional parameters for the model

        Returns:
            Pricing model instance
        """
        if model_type.lower() == 'black_scholes':
            return BlackScholes()
        elif model_type.lower() == 'monte_carlo':
            n_paths = kwargs.get('n_paths', 10000)
            random_seed = kwargs.get('random_seed', None)
            return MonteCarlo(n_paths=n_paths, random_seed=random_seed)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# ---- Usage Examples ----

def example_black_scholes():
    """Example of using the Black-Scholes model."""
    # Create option parameters
    params = OptionParams(
        S=100,  # Stock price
        K=100,  # Strike price
        T=1.0,  # Time to maturity (1 year)
        r=0.05,  # 5% risk-free rate
        sigma=0.2,  # 20% volatility
        option_type=OptionType.CALL
    )

    # Create model
    model = ModelFactory.create_model('black_scholes')

    # Calculate price and Greeks
    price = model.price(params)
    greeks = model.greeks(params)

    print(f"Black-Scholes Call Price: {price:.4f}")
    print(f"Greeks:")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")

    # Create analyzer and plot sensitivities
    analyzer = OptionAnalyzer(model)

    # Stock price sensitivity
    s_range = np.linspace(80, 120, 41)
    analyzer.plot_parameter_sensitivity(params, 'S', s_range)
    plt.show()

    # Volatility sensitivity
    sigma_range = np.linspace(0.1, 0.5, 41)
    analyzer.plot_parameter_sensitivity(params, 'sigma', sigma_range)
    plt.show()

    # Greeks sensitivity to stock price
    analyzer.plot_greeks(params, 'S', s_range)


def example_monte_carlo():
    """Example of using the Monte Carlo simulation."""
    # Create option parameters
    params = OptionParams(
        S=100,  # Stock price
        K=100,  # Strike price
        T=1.0,  # Time to maturity (1 year)
        r=0.05,  # 5% risk-free rate
        sigma=0.2,  # 20% volatility
        option_type=OptionType.CALL
    )

    # Create model
    model = ModelFactory.create_model('monte_carlo', n_paths=10000, random_seed=42)

    # Calculate price
    price = model.price(params)

    # Compare with Black-Scholes
    bs_model = ModelFactory.create_model('black_scholes')
    bs_price = bs_model.price(params)

    print(f"Monte Carlo Call Price: {price:.4f}")
    print(f"Black-Scholes Call Price: {bs_price:.4f}")
    print(f"Difference: {abs(price - bs_price):.4f}")


if __name__ == "__main__":
    print("Option Pricing Framework Examples")
    print("-" * 30)

    print("\nBlack-Scholes Example:")
    example_black_scholes()

    # print("\nMonte Carlo Example:")
    # example_monte_carlo()
