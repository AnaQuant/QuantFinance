"""
Base classes for stochastic processes.

This module defines abstract base classes that all stochastic process
implementations should inherit from.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class StochasticProcess(ABC):
    """
    Abstract base class for all stochastic processes.

    All stochastic processes should implement the simulate method and
    provide methods for computing theoretical moments when available.
    """

    def __init__(self, initial_value: float, random_seed: Optional[int] = None):
        """
        Initialize the stochastic process.

        Parameters
        ----------
        initial_value : float
            Initial value of the process X(0)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.initial_value = initial_value
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    @abstractmethod
    def simulate(
        self,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths of the stochastic process.

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
        pass

    def expected_value(self, t: float) -> float:
        """
        Compute the theoretical expected value E[X(t)] if known.

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        float
            Expected value at time t

        Raises
        ------
        NotImplementedError
            If theoretical expectation is not available
        """
        raise NotImplementedError(
            f"Theoretical expectation not implemented for {self.__class__.__name__}"
        )

    def variance(self, t: float) -> float:
        """
        Compute the theoretical variance Var[X(t)] if known.

        Parameters
        ----------
        t : float
            Time point

        Returns
        -------
        float
            Variance at time t

        Raises
        ------
        NotImplementedError
            If theoretical variance is not available
        """
        raise NotImplementedError(
            f"Theoretical variance not implemented for {self.__class__.__name__}"
        )


class DiscreteTimeModel(ABC):
    """
    Abstract base class for discrete-time models (e.g., tree models).
    """

    @abstractmethod
    def build_tree(self, n_steps: int, dt: float) -> np.ndarray:
        """
        Build the binomial/trinomial tree structure.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        dt : float
            Time step size

        Returns
        -------
        np.ndarray
            Tree structure
        """
        pass
