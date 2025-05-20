import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sco
from typing import List, Tuple, Dict, Optional, Union


class PortfolioOptimization:
    """
    Portfolio optimization class using GBM simulations for asset returns.
    """

    def __init__(self,
                 initial_prices: np.ndarray,
                 mu: np.ndarray,
                 sigma: np.ndarray,
                 corr_matrix: np.ndarray,
                 risk_free_rate: float = 0.0,
                 asset_names: Optional[List[str]] = None):
        """
        Initialize the portfolio optimization model.

        Parameters:
        -----------
        initial_prices: np.ndarray
            Initial prices for each asset
        mu: np.ndarray
            Expected returns (drift) for each asset
        sigma: np.ndarray
            Volatilities for each asset
        corr_matrix: np.ndarray
            Correlation matrix between assets
        risk_free_rate: float
            Risk-free rate for Sharpe ratio calculation
        asset_names: List[str], optional
            Names of the assets
        """
        self.initial_prices = initial_prices
        self.num_assets = len(initial_prices)
        self.mu = mu
        self.sigma = sigma
        self.corr_matrix = corr_matrix
        self.risk_free_rate = risk_free_rate

        # Compute the covariance matrix
        self.cov_matrix = np.outer(sigma, sigma) * corr_matrix

        # Asset names for plotting
        if asset_names is None:
            self.asset_names = [f"Asset {i + 1}" for i in range(self.num_assets)]
        else:
            self.asset_names = asset_names

    def simulate_gbm_returns(self, T: float = 1.0, dt: float = 0.01, num_paths: int = 1000,
                             seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate correlated asset returns using GBM.

        Parameters:
        -----------
        T: float
            Time horizon in years
        dt: float
            Time step size
        num_paths: int
            Number of simulation paths
        seed: int, optional
            Random seed for reproducibility

        Returns:
        --------
        np.ndarray
            Simulated returns for each asset and path
        """
        if seed is not None:
            np.random.seed(seed)

        steps = int(T / dt)

        # Generate correlated normal random variables
        Z = np.random.standard_normal((steps, self.num_assets, num_paths))

        # Use Cholesky decomposition for correlation
        L = np.linalg.cholesky(self.corr_matrix)

        # Correlate the random variables
        correlated_Z = np.zeros_like(Z)
        for t in range(steps):
            correlated_Z[t] = np.dot(L, Z[t])

        # Initialize price paths
        paths = np.zeros((steps + 1, self.num_assets, num_paths))
        paths[0] = self.initial_prices[:, np.newaxis]

        # Simulate price paths
        for t in range(1, steps + 1):
            # GBM formula: S(t) = S(0) * exp((mu - 0.5 * sigma^2) * t + sigma * W(t))
            drift = (self.mu - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma[:, np.newaxis] * correlated_Z[t - 1] * np.sqrt(dt)
            paths[t] = paths[t - 1] * np.exp(drift[:, np.newaxis] + diffusion)

        # Calculate total returns for the period
        returns = (paths[-1] / paths[0]) - 1

        return returns

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics.

        Parameters:
        -----------
        weights: np.ndarray
            Asset weights in the portfolio

        Returns:
        --------
        Tuple[float, float, float]
            Expected return, volatility, and Sharpe ratio
        """
        # Expected portfolio return
        port_return = np.sum(weights * self.mu)

        # Expected portfolio volatility
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol

        return port_return, port_vol, sharpe

    def negative_sharpe(self, weights: np.ndarray) -> float:
        """
        Calculate negative Sharpe ratio for minimization.

        Parameters:
        -----------
        weights: np.ndarray
            Asset weights in the portfolio

        Returns:
        --------
        float
            Negative Sharpe ratio
        """
        return -self.portfolio_performance(weights)[2]

    def optimize_portfolio(self, constraints: Optional[List] = None) -> Dict:
        """
        Find the optimal portfolio weights to maximize Sharpe ratio.

        Parameters:
        -----------
        constraints: List, optional
            Additional constraints for the optimization

        Returns:
        --------
        Dict
            Optimization results
        """
        # Initial guess (equal weights)
        x0 = np.ones(self.num_assets) / self.num_assets

        # Constraint: sum of weights = 1
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Add any additional constraints
        if constraints is not None:
            constraints_list.extend(constraints)

        # Bounds: no short-selling (weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(self.num_assets))

        # Optimize
        result = sco.minimize(self.negative_sharpe, x0,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=constraints_list)

        # Extract the optimal weights
        optimal_weights = result['x']

        # Calculate portfolio performance
        expected_return, volatility, sharpe = self.portfolio_performance(optimal_weights)

        return {
            'weights': optimal_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'success': result['success'],
            'status': result['message']
        }

    def monte_carlo_portfolios(self, num_portfolios: int = 10000, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate random portfolios for Monte Carlo simulation.

        Parameters:
        -----------
        num_portfolios: int
            Number of random portfolios to generate
        seed: int, optional
            Random seed for reproducibility

        Returns:
        --------
        pd.DataFrame
            DataFrame with portfolio metrics
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize arrays
        results = np.zeros((num_portfolios, self.num_assets + 3))

        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)

            # Calculate portfolio performance
            port_return, port_vol, sharpe = self.portfolio_performance(weights)

            # Store results
            results[i, :self.num_assets] = weights
            results[i, self.num_assets] = port_return
            results[i, self.num_assets + 1] = port_vol
            results[i, self.num_assets + 2] = sharpe

        # Create column names
        columns = self.asset_names + ['Return', 'Volatility', 'Sharpe']

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=columns)

        return df

    def plot_efficient_frontier(self, num_portfolios: int = 10000,
                                highlight_optimal: bool = True,
                                highlight_assets: bool = True,
                                seed: Optional[int] = None):
        """
        Plot the efficient frontier with random portfolios.

        Parameters:
        -----------
        num_portfolios: int
            Number of random portfolios to generate
        highlight_optimal: bool
            Whether to highlight the optimal portfolio
        highlight_assets: bool
            Whether to highlight individual assets
        seed: int, optional
            Random seed for reproducibility
        """
        # Generate random portfolios
        portfolios = self.monte_carlo_portfolios(num_portfolios, seed)

        # Optimize portfolio
        if highlight_optimal:
            optimal = self.optimize_portfolio()

        # Plot
        plt.figure(figsize=(12, 8))

        # Plot random portfolios
        plt.scatter(portfolios['Volatility'], portfolios['Return'],
                    c=portfolios['Sharpe'], cmap='viridis',
                    marker='o', s=10, alpha=0.3)

        # Highlight individual assets if requested
        if highlight_assets:
            # Calculate metrics for individual assets
            for i in range(self.num_assets):
                weights = np.zeros(self.num_assets)
                weights[i] = 1.0
                ret, vol, _ = self.portfolio_performance(weights)
                plt.scatter(vol, ret, marker='*', color='red', s=100)
                plt.annotate(self.asset_names[i], (vol, ret),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=10)

        # Highlight optimal portfolio if requested
        if highlight_optimal:
            plt.scatter(optimal['volatility'], optimal['expected_return'],
                        marker='*', color='green', s=200)
            plt.annotate('Optimal Portfolio', (optimal['volatility'], optimal['expected_return']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=12)

        plt.colorbar(label='Sharpe Ratio')
        plt.title('Portfolio Optimization - Efficient Frontier')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.grid(True)
        plt.show()

    def display_optimal_allocation(self):
        """Display the optimal portfolio allocation."""
        optimal = self.optimize_portfolio()

        print(f"Optimal Portfolio:")
        print(f"Expected Annual Return: {optimal['expected_return'] * 100:.2f}%")
        print(f"Expected Annual Volatility: {optimal['volatility'] * 100:.2f}%")
        print(f"Sharpe Ratio: {optimal['sharpe_ratio']:.4f}")
        print("\nOptimal Asset Allocation:")

        # Display as a DataFrame
        allocation_df = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': optimal['weights'] * 100
        })
        allocation_df = allocation_df.sort_values(by='Weight', ascending=False)
        allocation_df['Weight'] = allocation_df['Weight'].apply(lambda x: f"{x:.2f}%")

        print(allocation_df)

    def run_gbm_simulation_for_portfolio(self, weights: np.ndarray,
                                         T: float = 1.0, dt: float = 0.01,
                                         num_paths: int = 1000, seed: Optional[int] = None) -> Dict:
        """
        Run GBM simulation for a specific portfolio allocation.

        Parameters:
        -----------
        weights: np.ndarray
            Portfolio weights
        T: float
            Time horizon in years
        dt: float
            Time step size
        num_paths: int
            Number of simulation paths
        seed: int, optional
            Random seed for reproducibility

        Returns:
        --------
        Dict
            Simulation results
        """
        # Simulate asset returns
        asset_returns = self.simulate_gbm_returns(T, dt, num_paths, seed)

        # Calculate portfolio returns
        portfolio_returns = np.sum(asset_returns * weights[:, np.newaxis], axis=0)

        # Calculate statistics
        mean_return = np.mean(portfolio_returns)
        median_return = np.median(portfolio_returns)
        std_dev = np.std(portfolio_returns)
        var_95 = np.percentile(portfolio_returns, 5)  # 95% VaR
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()  # 95% CVaR

        return {
            'portfolio_returns': portfolio_returns,
            'mean_return': mean_return,
            'median_return': median_return,
            'std_dev': std_dev,
            'var_95': var_95,
            'cvar_95': cvar_95
        }

    def plot_portfolio_returns_distribution(self, weights: Optional[np.ndarray] = None,
                                            T: float = 1.0, dt: float = 0.01,
                                            num_paths: int = 10000, seed: Optional[int] = None):
        """
        Plot the distribution of portfolio returns.

        Parameters:
        -----------
        weights: np.ndarray, optional
            Portfolio weights (if None, use optimal weights)
        T: float
            Time horizon in years
        dt: float
            Time step size
        num_paths: int
            Number of simulation paths
        seed: int, optional
            Random seed for reproducibility
        """
        if weights is None:
            # Use optimal weights if none provided
            weights = self.optimize_portfolio()['weights']

        # Run simulation
        results = self.run_gbm_simulation_for_portfolio(weights, T, dt, num_paths, seed)

        # Plot distribution
        plt.figure(figsize=(12, 8))

        # Histogram and KDE
        sns.histplot(results['portfolio_returns'], kde=True, stat='density')

        # Add lines for key statistics
        plt.axvline(results['mean_return'], color='g', linestyle='-',
                    label=f'Mean: {results["mean_return"] * 100:.2f}%')
        plt.axvline(results['var_95'], color='r', linestyle='--',
                    label=f'95% VaR: {results["var_95"] * 100:.2f}%')
        plt.axvline(results['cvar_95'], color='darkred', linestyle=':',
                    label=f'95% CVaR: {results["cvar_95"] * 100:.2f}%')

        plt.title(f'Portfolio Returns Distribution (T={T} years, {num_paths} simulations)')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        return results


# Example usage
if __name__ == "__main__":
    # Example with 4 assets
    initial_prices = np.array([100.0, 150.0, 200.0, 50.0])
    mu = np.array([0.05, 0.08, 0.12, 0.03])  # Expected returns
    sigma = np.array([0.15, 0.22, 0.30, 0.10])  # Volatilities

    # Correlation matrix
    corr_matrix = np.array([
        [1.00, 0.25, 0.10, 0.50],
        [0.25, 1.00, 0.35, 0.20],
        [0.10, 0.35, 1.00, 0.15],
        [0.50, 0.20, 0.15, 1.00]
    ])

    # Asset names
    asset_names = ["Stock A", "Stock B", "Stock C", "Bond D"]

    # Create portfolio optimizer
    portfolio_opt = PortfolioOptimization(
        initial_prices=initial_prices,
        mu=mu,
        sigma=sigma,
        corr_matrix=corr_matrix,
        risk_free_rate=0.01,
        asset_names=asset_names
    )

    # Display optimal allocation
    portfolio_opt.display_optimal_allocation()

    # Plot efficient frontier
    portfolio_opt.plot_efficient_frontier(num_portfolios=5000, seed=42)

    # Plot portfolio returns distribution
    portfolio_opt.plot_portfolio_returns_distribution(T=1.0, num_paths=10000, seed=42)
