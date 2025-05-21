import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import List, Dict, Union, Tuple, Optional, Callable
import warnings

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")
warnings.filterwarnings('ignore')


class Instrument:
    """Base class for financial instruments."""

    def __init__(self, name: str, ticker: str):
        self.name = name
        self.ticker = ticker

    def get_price(self, scenario_prices: Dict[str, float]) -> float:
        """Get the price of the instrument under a given scenario."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_delta(self) -> float:
        """Get the delta of the instrument."""
        raise NotImplementedError("Subclasses must implement this method")


class Equity(Instrument):
    """Class representing an equity position."""

    def __init__(self, name: str, ticker: str, quantity: float, current_price: float,
                 annual_dividend_yield: float = 0.0):
        super().__init__(name, ticker)
        self.quantity = quantity
        self.current_price = current_price
        self.annual_dividend_yield = annual_dividend_yield

    def get_price(self, scenario_prices: Dict[str, float]) -> float:
        """Get the price of the equity under a given scenario."""
        if self.ticker in scenario_prices:
            return scenario_prices[self.ticker] * self.quantity
        return self.current_price * self.quantity

    def get_delta(self) -> float:
        """Get the delta of the equity."""
        return self.quantity

    def __str__(self) -> str:
        return f"{self.name} (Equity): {self.quantity} shares @ ${self.current_price:.2f}"


class VanillaOption(Instrument):
    """Class representing a vanilla option position."""

    def __init__(self, name: str, underlying_ticker: str, option_type: str,
                 strike: float, expiry_date: datetime, quantity: float,
                 current_price: float, underlying_price: float,
                 risk_free_rate: float = 0.02, volatility: float = 0.2):
        super().__init__(name, f"{underlying_ticker}_OPT")
        self.underlying_ticker = underlying_ticker
        self.option_type = option_type.lower()  # 'call' or 'put'
        self.strike = strike
        self.expiry_date = expiry_date
        self.quantity = quantity
        self.current_price = current_price
        self.underlying_price = underlying_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")

    def time_to_expiry(self, valuation_date: Optional[datetime] = None) -> float:
        """Calculate time to expiry in years."""
        if valuation_date is None:
            valuation_date = datetime.now()
        days_to_expiry = (self.expiry_date - valuation_date).days
        return max(0, days_to_expiry / 365.0)

    def black_scholes_price(self, S: float, T: float) -> float:
        """Calculate Black-Scholes option price."""
        K = self.strike
        r = self.risk_free_rate
        sigma = self.volatility

        if T <= 0:
            # Option is expired
            if self.option_type == 'call':
                return max(0, S - K) * self.quantity
            else:
                return max(0, K - S) * self.quantity

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == 'call':
            return (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)) * self.quantity
        else:  # put
            return (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)) * self.quantity

    def get_price(self, scenario_prices: Dict[str, float], valuation_date: Optional[datetime] = None) -> float:
        """Get the price of the option under a given scenario."""
        if self.underlying_ticker in scenario_prices:
            S = scenario_prices[self.underlying_ticker]
        else:
            S = self.underlying_price

        T = self.time_to_expiry(valuation_date)
        return self.black_scholes_price(S, T)

    def get_delta(self, valuation_date: Optional[datetime] = None) -> float:
        """Get the delta of the option."""
        S = self.underlying_price
        T = self.time_to_expiry(valuation_date)

        if T <= 0:
            if self.option_type == 'call':
                return 1.0 * self.quantity if S > self.strike else 0.0
            else:
                return -1.0 * self.quantity if S < self.strike else 0.0

        sigma = self.volatility
        d1 = (np.log(S / self.strike) + (self.risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if self.option_type == 'call':
            return norm.cdf(d1) * self.quantity
        else:  # put
            return (norm.cdf(d1) - 1) * self.quantity

    def get_gamma(self, valuation_date: Optional[datetime] = None) -> float:
        """Get the gamma of the option."""
        S = self.underlying_price
        T = self.time_to_expiry(valuation_date)

        if T <= 0:
            return 0.0

        sigma = self.volatility
        d1 = (np.log(S / self.strike) + (self.risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma * self.quantity

    def get_vega(self, valuation_date: Optional[datetime] = None) -> float:
        """Get the vega of the option (sensitivity to volatility)."""
        S = self.underlying_price
        T = self.time_to_expiry(valuation_date)

        if T <= 0:
            return 0.0

        sigma = self.volatility
        d1 = (np.log(S / self.strike) + (self.risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01  # 1% change in volatility
        return vega * self.quantity

    def get_theta(self, valuation_date: Optional[datetime] = None) -> float:
        """Get the theta of the option (sensitivity to time)."""
        S = self.underlying_price
        T = self.time_to_expiry(valuation_date)

        if T <= 0:
            return 0.0

        sigma = self.volatility
        r = self.risk_free_rate
        K = self.strike

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if self.option_type == 'call':
            theta -= r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            theta += r * K * np.exp(-r * T) * norm.cdf(-d2)

        return theta * self.quantity / 365.0  # Convert to daily theta

    def get_rho(self, valuation_date: Optional[datetime] = None) -> float:
        """Get the rho of the option (sensitivity to interest rate)."""
        S = self.underlying_price
        T = self.time_to_expiry(valuation_date)

        if T <= 0:
            return 0.0

        sigma = self.volatility
        r = self.risk_free_rate
        K = self.strike

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01  # 1% change in interest rate
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

        return rho * self.quantity

    def __str__(self) -> str:
        option_type_str = "Call" if self.option_type == 'call' else "Put"
        days_to_expiry = (self.expiry_date - datetime.now()).days
        return (f"{self.name} ({option_type_str}): {self.quantity} contracts, "
                f"K=${self.strike:.2f}, {days_to_expiry} days to expiry, "
                f"premium=${self.current_price:.2f}")


class Portfolio:
    """Class representing a portfolio of financial instruments."""

    def __init__(self, name: str):
        self.name = name
        self.instruments: List[Instrument] = []

    def add_instrument(self, instrument: Instrument) -> None:
        """Add an instrument to the portfolio."""
        self.instruments.append(instrument)

    def get_current_value(self) -> float:
        """Calculate the current value of the portfolio."""
        return sum(
            instrument.get_price({}) for instrument in self.instruments
        )

    def get_value_under_scenario(self, scenario_prices: Dict[str, float],
                                 valuation_date: Optional[datetime] = None) -> float:
        """Calculate the portfolio value under a given price scenario."""
        return sum(
            instrument.get_price(scenario_prices, valuation_date)
            for instrument in self.instruments
        )

    def get_exposure_by_ticker(self) -> Dict[str, float]:
        """Calculate the portfolio exposure by ticker."""
        exposures = {}
        for instrument in self.instruments:
            if isinstance(instrument, Equity):
                ticker = instrument.ticker
                exposure = instrument.current_price * instrument.quantity
            elif isinstance(instrument, VanillaOption):
                ticker = instrument.underlying_ticker
                exposure = instrument.get_delta() * instrument.underlying_price
            else:
                continue

            if ticker in exposures:
                exposures[ticker] += exposure
            else:
                exposures[ticker] = exposure

        return exposures

    def get_delta_by_ticker(self) -> Dict[str, float]:
        """Calculate the portfolio delta by ticker."""
        deltas = {}
        for instrument in self.instruments:
            if isinstance(instrument, Equity):
                ticker = instrument.ticker
                delta = instrument.get_delta()
            elif isinstance(instrument, VanillaOption):
                ticker = instrument.underlying_ticker
                delta = instrument.get_delta()
            else:
                continue

            if ticker in deltas:
                deltas[ticker] += delta
            else:
                deltas[ticker] = delta

        return deltas

    def get_instruments_by_type(self) -> Dict[str, List[Instrument]]:
        """Group instruments by type."""
        instruments_by_type = {}
        for instrument in self.instruments:
            instrument_type = instrument.__class__.__name__
            if instrument_type not in instruments_by_type:
                instruments_by_type[instrument_type] = []
            instruments_by_type[instrument_type].append(instrument)

        return instruments_by_type

    def summary(self) -> pd.DataFrame:
        """Create a summary of the portfolio."""
        data = []

        for instrument in self.instruments:
            if isinstance(instrument, Equity):
                data.append({
                    'Type': 'Equity',
                    'Name': instrument.name,
                    'Ticker': instrument.ticker,
                    'Quantity': instrument.quantity,
                    'Price': instrument.current_price,
                    'Value': instrument.current_price * instrument.quantity,
                    'Delta': instrument.get_delta()
                })
            elif isinstance(instrument, VanillaOption):
                option_type = "Call" if instrument.option_type == 'call' else "Put"
                days_to_expiry = (instrument.expiry_date - datetime.now()).days
                data.append({
                    'Type': f'{option_type} Option',
                    'Name': instrument.name,
                    'Ticker': instrument.underlying_ticker,
                    'Quantity': instrument.quantity,
                    'Strike': instrument.strike,
                    'Days to Expiry': days_to_expiry,
                    'Price': instrument.current_price,
                    'Value': instrument.current_price * instrument.quantity,
                    'Delta': instrument.get_delta(),
                    'Gamma': instrument.get_gamma(),
                    'Vega': instrument.get_vega(),
                    'Theta': instrument.get_theta(),
                    'Rho': instrument.get_rho()
                })

        return pd.DataFrame(data)

    def __str__(self) -> str:
        summary = f"{self.name} Portfolio\n"
        summary += f"Total Value: ${self.get_current_value():.2f}\n"
        summary += f"Number of Instruments: {len(self.instruments)}\n"
        summary += "Instruments:\n"

        for instrument in self.instruments:
            summary += f"  - {instrument}\n"

        return summary


class RiskAnalyzer:
    """Class for analyzing risk metrics for a portfolio."""

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.scenarios = {}
        self.base_scenario = {}

    def set_base_scenario(self, scenario_prices: Dict[str, float]) -> None:
        """Set the base scenario for risk analysis."""
        self.base_scenario = scenario_prices

    def add_scenario(self, name: str, scenario_prices: Dict[str, float]) -> None:
        """Add a scenario for risk analysis."""
        self.scenarios[name] = scenario_prices

    def add_stress_scenario(self, name: str, stressed_tickers: List[str],
                            stress_factor: float) -> None:
        """Add a stress scenario based on percentage changes."""
        stress_scenario = self.base_scenario.copy()

        for ticker in stressed_tickers:
            if ticker in stress_scenario:
                stress_scenario[ticker] *= (1 + stress_factor)

        self.scenarios[name] = stress_scenario

    def calculate_value_at_risk(self, confidence_level: float = 0.95,
                                time_horizon: int = 1,
                                num_simulations: int = 10000,
                                correlation_matrix: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using Monte Carlo simulation.

        Parameters:
        -----------
        confidence_level: float
            Confidence level for VaR calculation (e.g., 0.95 for 95%)
        time_horizon: int
            Time horizon in days
        num_simulations: int
            Number of simulations to run
        correlation_matrix: pd.DataFrame, optional
            Correlation matrix between tickers

        Returns:
        --------
        Dict[str, float]
            Dictionary containing VaR metrics
        """
        # Get unique tickers in the portfolio
        tickers = set()
        for instrument in self.portfolio.instruments:
            if isinstance(instrument, Equity):
                tickers.add(instrument.ticker)
            elif isinstance(instrument, VanillaOption):
                tickers.add(instrument.underlying_ticker)

        tickers = list(tickers)

        # Get current prices
        current_prices = {}
        for ticker in tickers:
            for instrument in self.portfolio.instruments:
                if isinstance(instrument, Equity) and instrument.ticker == ticker:
                    current_prices[ticker] = instrument.current_price
                    break
                elif isinstance(instrument, VanillaOption) and instrument.underlying_ticker == ticker:
                    current_prices[ticker] = instrument.underlying_price
                    break

        # Get volatilities (simple approach - can be improved with actual market data)
        volatilities = {}
        for ticker in tickers:
            for instrument in self.portfolio.instruments:
                if isinstance(instrument, VanillaOption) and instrument.underlying_ticker == ticker:
                    volatilities[ticker] = instrument.volatility
                    break
            if ticker not in volatilities:
                volatilities[ticker] = 0.2  # Default volatility

        # Create correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = pd.DataFrame(np.identity(len(tickers)), index=tickers, columns=tickers)

        # Convert to daily volatility
        daily_volatilities = {ticker: vol / np.sqrt(252) for ticker, vol in volatilities.items()}

        # Generate random returns with correlation
        np.random.seed(42)
        random_returns = np.random.normal(0, 1, size=(len(tickers), num_simulations))

        # Apply Cholesky decomposition for correlated returns
        cholesky = np.linalg.cholesky(correlation_matrix.values)
        correlated_returns = np.dot(cholesky, random_returns)

        # Apply daily volatility and calculate simulated prices
        simulated_returns = {}
        for i, ticker in enumerate(tickers):
            daily_vol = daily_volatilities[ticker]
            adjusted_returns = correlated_returns[i] * daily_vol * np.sqrt(time_horizon)
            simulated_returns[ticker] = adjusted_returns

        # Calculate portfolio values under simulated scenarios
        portfolio_values = []
        base_value = self.portfolio.get_current_value()

        for sim in range(num_simulations):
            scenario_prices = {}
            for ticker in tickers:
                if ticker in current_prices:
                    sim_return = simulated_returns[ticker][sim]
                    scenario_prices[ticker] = current_prices[ticker] * np.exp(sim_return)

            portfolio_value = self.portfolio.get_value_under_scenario(scenario_prices)
            portfolio_values.append(portfolio_value)

        # Calculate VaR metrics
        pnl_changes = np.array(portfolio_values) - base_value
        var_absolute = np.percentile(pnl_changes, (1 - confidence_level) * 100)
        cvar_absolute = pnl_changes[pnl_changes <= var_absolute].mean()

        var_percentage = var_absolute / base_value
        cvar_percentage = cvar_absolute / base_value

        return {
            'var_absolute': var_absolute,
            'cvar_absolute': cvar_absolute,
            'var_percentage': var_percentage,
            'cvar_percentage': cvar_percentage,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'base_value': base_value
        }

    def calculate_scenario_pnl(self, valuation_date: Optional[datetime] = None) -> pd.DataFrame:
        """Calculate PnL under different scenarios."""
        base_value = self.portfolio.get_value_under_scenario(self.base_scenario, valuation_date)

        results = []
        for name, scenario in self.scenarios.items():
            scenario_value = self.portfolio.get_value_under_scenario(scenario, valuation_date)
            pnl = scenario_value - base_value
            pnl_percentage = pnl / base_value * 100

            results.append({
                'Scenario': name,
                'Base Value': base_value,
                'Scenario Value': scenario_value,
                'PnL': pnl,
                'PnL %': pnl_percentage
            })

        return pd.DataFrame(results)

    def plot_scenario_analysis(self, valuation_date: Optional[datetime] = None) -> None:
        """Plot the results of scenario analysis."""
        scenario_pnl = self.calculate_scenario_pnl(valuation_date)

        plt.figure(figsize=(12, 6))

        # Sort by PnL for better visualization
        scenario_pnl = scenario_pnl.sort_values('PnL')

        # Plot absolute PnL
        ax = sns.barplot(x='Scenario', y='PnL', data=scenario_pnl)

        # Add value labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'${p.get_height():.2f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

        plt.title('Scenario Analysis - Absolute PnL')
        plt.xlabel('Scenario')
        plt.ylabel('PnL ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Plot percentage PnL
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Scenario', y='PnL %', data=scenario_pnl)

        # Add value labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_height():.2f}%',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')

        plt.title('Scenario Analysis - Percentage PnL')
        plt.xlabel('Scenario')
        plt.ylabel('PnL (%)')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_var_distribution(self, confidence_level: float = 0.95,
                              time_horizon: int = 1,
                              num_simulations: int = 10000) -> None:
        """Plot the VaR distribution."""
        var_results = self.calculate_value_at_risk(confidence_level, time_horizon, num_simulations)

        # Get unique tickers in the portfolio
        tickers = set()
        for instrument in self.portfolio.instruments:
            if isinstance(instrument, Equity):
                tickers.add(instrument.ticker)
            elif isinstance(instrument, VanillaOption):
                tickers.add(instrument.underlying_ticker)

        tickers = list(tickers)

        # Get current prices
        current_prices = {}
        for ticker in tickers:
            for instrument in self.portfolio.instruments:
                if isinstance(instrument, Equity) and instrument.ticker == ticker:
                    current_prices[ticker] = instrument.current_price
                    break
                elif isinstance(instrument, VanillaOption) and instrument.underlying_ticker == ticker:
                    current_prices[ticker] = instrument.underlying_price
                    break

        # Get volatilities
        volatilities = {}
        for ticker in tickers:
            for instrument in self.portfolio.instruments:
                if isinstance(instrument, VanillaOption) and instrument.underlying_ticker == ticker:
                    volatilities[ticker] = instrument.volatility
                    break
            if ticker not in volatilities:
                volatilities[ticker] = 0.2  # Default volatility

        # Create correlation matrix
        correlation_matrix = pd.DataFrame(np.identity(len(tickers)), index=tickers, columns=tickers)

        # Convert to daily volatility
        daily_volatilities = {ticker: vol / np.sqrt(252) for ticker, vol in volatilities.items()}

        # Generate random returns with correlation
        np.random.seed(42)
        random_returns = np.random.normal(0, 1, size=(len(tickers), num_simulations))

        # Apply Cholesky decomposition for correlated returns
        cholesky = np.linalg.cholesky(correlation_matrix.values)
        correlated_returns = np.dot(cholesky, random_returns)

        # Apply daily volatility and calculate simulated prices
        simulated_returns = {}
        for i, ticker in enumerate(tickers):
            daily_vol = daily_volatilities[ticker]
            adjusted_returns = correlated_returns[i] * daily_vol * np.sqrt(time_horizon)
            simulated_returns[ticker] = adjusted_returns

        # Calculate portfolio values under simulated scenarios
        portfolio_values = []
        base_value = self.portfolio.get_current_value()

        for sim in range(num_simulations):
            scenario_prices = {}
            for ticker in tickers:
                if ticker in current_prices:
                    sim_return = simulated_returns[ticker][sim]
                    scenario_prices[ticker] = current_prices[ticker] * np.exp(sim_return)

            portfolio_value = self.portfolio.get_value_under_scenario(scenario_prices)
            portfolio_values.append(portfolio_value)

        # Calculate PnL changes
        pnl_changes = np.array(portfolio_values) - base_value

        # Plot distribution
        plt.figure(figsize=(12, 6))

        sns.histplot(pnl_changes, kde=True, stat='density')

        # Add VaR line
        plt.axvline(x=var_results['var_absolute'], color='red', linestyle='--',
                    label=f"{confidence_level * 100}% VaR: ${abs(var_results['var_absolute']):.2f}")

        # Add CVaR line
        plt.axvline(x=var_results['cvar_absolute'], color='darkred', linestyle=':',
                    label=f"{confidence_level * 100}% CVaR: ${abs(var_results['cvar_absolute']):.2f}")

        plt.title(f'Portfolio PnL Distribution ({time_horizon} Day Horizon)')
        plt.xlabel('PnL ($)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_sensitivity_analysis(self, ticker: str,
                                 price_range: List[float],
                                 valuation_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Run sensitivity analysis for a specific ticker.

        Parameters:
        -----------
        ticker: str
            The ticker to analyze
        price_range: List[float]
            List of price levels to test
        valuation_date: datetime, optional
            Valuation date for the analysis

        Returns:
        --------
        pd.DataFrame
            DataFrame with sensitivity analysis results
        """
        results = []
        base_price = None

        # Find the base price for the ticker
        for instrument in self.portfolio.instruments:
            if isinstance(instrument, Equity) and instrument.ticker == ticker:
                base_price = instrument.current_price
                break
            elif isinstance(instrument, VanillaOption) and instrument.underlying_ticker == ticker:
                base_price = instrument.underlying_price
                break

        if base_price is None:
            raise ValueError(f"Ticker {ticker} not found in portfolio")

        # Calculate portfolio value at different price levels
        for price in price_range:
            scenario_prices = {ticker: price}
            portfolio_value = self.portfolio.get_value_under_scenario(scenario_prices, valuation_date)

            price_change = price / base_price - 1

            results.append({
                'Price': price,
                'Price Change': price_change,
                'Portfolio Value': portfolio_value
            })

        return pd.DataFrame(results)

    def plot_sensitivity_analysis(self, ticker: str,
                                  price_range_percent: List[float],
                                  valuation_date: Optional[datetime] = None) -> None:
        """
        Plot sensitivity analysis for a specific ticker.

        Parameters:
        -----------
        ticker: str
            The ticker to analyze
        price_range_percent: List[float]
            List of percentage price changes to test
        valuation_date: datetime, optional
            Valuation date for the analysis
        """
        # Find the base price for the ticker
        base_price = None
        for instrument in self.portfolio.instruments:
            if isinstance(instrument, Equity) and instrument.ticker == ticker:
                base_price = instrument.current_price
                break
            elif isinstance(instrument, VanillaOption) and instrument.underlying_ticker == ticker:
                base_price = instrument.underlying_price
                break

        if base_price is None:
            raise ValueError(f"Ticker {ticker} not found in portfolio")

        # Convert percentage changes to absolute prices
        price_range = [base_price * (1 + pct / 100) for pct in price_range_percent]

        # Run sensitivity analysis
        results = self.run_sensitivity_analysis(ticker, price_range, valuation_date)

        # Calculate base portfolio value
        base_portfolio_value = self.portfolio.get_current_value()

        # Add PnL columns
        results['PnL'] = results['Portfolio Value'] - base_portfolio_value
        results['PnL %'] = results['PnL'] / base_portfolio_value * 100

        # Plot results
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        plt.plot(results['Price Change'] * 100, results['Portfolio Value'], marker='o')
        plt.axhline(y=base_portfolio_value, color='red', linestyle='--',
                    label=f'Base Value: ${base_portfolio_value:.2f}')
        plt.title(f'Portfolio Sensitivity to {ticker} Price Changes')
        plt