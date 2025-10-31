"""
Utility functions for stochastic process simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


def generate_brownian_increments(
    n_steps: int,
    n_paths: int,
    dt: float,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Brownian motion increments.

    Parameters
    ----------
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths
    dt : float
        Time step size
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Brownian increments of shape (n_paths, n_steps)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.sqrt(dt) * np.random.randn(n_paths, n_steps)


def plot_paths(
    t: np.ndarray,
    paths: np.ndarray,
    title: str = "Simulated Paths",
    xlabel: str = "Time",
    ylabel: str = "Value",
    n_paths_to_plot: Optional[int] = None,
    show_mean: bool = True,
    show_confidence_bands: bool = False,
    alpha: float = 0.1,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot simulated paths of a stochastic process.

    Parameters
    ----------
    t : np.ndarray
        Time grid
    paths : np.ndarray
        Simulated paths of shape (n_paths, n_steps)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    n_paths_to_plot : int, optional
        Number of paths to plot (default: min(100, total paths))
    show_mean : bool
        Whether to show the mean path
    show_confidence_bands : bool
        Whether to show confidence bands (±1 std)
    alpha : float
        Transparency of individual paths
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)

    n_paths = paths.shape[0]
    if n_paths_to_plot is None:
        n_paths_to_plot = min(100, n_paths)

    # Plot individual paths
    for i in range(n_paths_to_plot):
        plt.plot(t, paths[i, :], color='blue', alpha=alpha, linewidth=0.5)

    # Plot mean
    if show_mean:
        mean_path = np.mean(paths, axis=0)
        plt.plot(t, mean_path, color='red', linewidth=2, label='Mean', linestyle='--')

    # Plot confidence bands
    if show_confidence_bands:
        mean_path = np.mean(paths, axis=0)
        std_path = np.std(paths, axis=0)
        plt.fill_between(
            t,
            mean_path - std_path,
            mean_path + std_path,
            color='red',
            alpha=0.2,
            label='±1 Std Dev'
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if show_mean or show_confidence_bands:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_empirical_theoretical(
    t: np.ndarray,
    paths: np.ndarray,
    theoretical_mean: Optional[np.ndarray] = None,
    theoretical_std: Optional[np.ndarray] = None,
    figsize: tuple = (14, 5)
) -> None:
    """
    Compare empirical statistics from simulations with theoretical values.

    Parameters
    ----------
    t : np.ndarray
        Time grid
    paths : np.ndarray
        Simulated paths of shape (n_paths, n_steps)
    theoretical_mean : np.ndarray, optional
        Theoretical mean at each time point
    theoretical_std : np.ndarray, optional
        Theoretical standard deviation at each time point
    figsize : tuple
        Figure size
    """
    empirical_mean = np.mean(paths, axis=0)
    empirical_std = np.std(paths, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Mean comparison
    ax1.plot(t, empirical_mean, label='Empirical Mean', linewidth=2)
    if theoretical_mean is not None:
        ax1.plot(t, theoretical_mean, label='Theoretical Mean',
                linestyle='--', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean')
    ax1.set_title('Mean Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Standard deviation comparison
    ax2.plot(t, empirical_std, label='Empirical Std Dev', linewidth=2)
    if theoretical_std is not None:
        ax2.plot(t, theoretical_std, label='Theoretical Std Dev',
                linestyle='--', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Standard Deviation Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_distribution(
    paths: np.ndarray,
    time_index: int = -1,
    bins: int = 50,
    theoretical_pdf: Optional[callable] = None,
    x_range: Optional[tuple] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot the distribution of process values at a specific time point.

    Parameters
    ----------
    paths : np.ndarray
        Simulated paths of shape (n_paths, n_steps)
    time_index : int
        Time index to plot (default: -1, i.e., final time)
    bins : int
        Number of histogram bins
    theoretical_pdf : callable, optional
        Function that returns theoretical PDF values given x values
    x_range : tuple, optional
        Range for x-axis (min, max)
    figsize : tuple
        Figure size
    """
    values = paths[:, time_index]

    plt.figure(figsize=figsize)
    plt.hist(values, bins=bins, density=True, alpha=0.6,
            color='blue', label='Empirical Distribution')

    if theoretical_pdf is not None:
        if x_range is None:
            x_min, x_max = values.min(), values.max()
        else:
            x_min, x_max = x_range
        x_vals = np.linspace(x_min, x_max, 200)
        y_vals = theoretical_pdf(x_vals)
        plt.plot(x_vals, y_vals, 'r-', linewidth=2,
                label='Theoretical Distribution')

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title(f'Distribution at Time Index {time_index}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
