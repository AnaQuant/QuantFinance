"""
Options Pricing Framework
A modular framework for options pricing with real market data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import requests
import yfinance as yf
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy import optimize
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('options_pricer')


# ========================
# DATA PROVIDER FRAMEWORK
# ========================

class DataProvider(ABC):
    """Abstract base class for all data providers"""

    @abstractmethod
    def get_historical_prices(self, ticker: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """Get historical price data for a given ticker"""
        pass

    @abstractmethod
    def get_options_chain(self, ticker: str, expiry_date: Optional[dt.datetime] = None) -> pd.DataFrame:
        """Get options chain data for a given ticker and expiry date"""
        pass

    @abstractmethod
    def get_risk_free_rate(self, term: float) -> float:
        """Get risk-free rate for a given term in years"""
        pass


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider implementation"""

    def get_historical_prices(self, ticker: str, start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
        """Get historical price data from Yahoo Finance"""
        logger.info(f"Fetching historical prices for {ticker} from {start_date} to {end_date}")

        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()

            # Keep only required columns
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            return data

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def get_options_chain(self, ticker: str, expiry_date: Optional[dt.datetime] = None) -> pd.DataFrame:
        """Get options chain from Yahoo Finance"""
        logger.info(f"Fetching options chain for {ticker}, expiry: {expiry_date}")

        try:
            stock = yf.Ticker(ticker)

            # Get all available expiry dates if none provided
            if expiry_date is None:
                expirations = stock.options
                if not expirations:
                    logger.warning(f"No options data available for {ticker}")
                    return pd.DataFrame()

                # Use the nearest expiration date
                expiry_date = dt.datetime.strptime(expirations[0], '%Y-%m-%d')

            # Format date for Yahoo Finance
            date_str = expiry_date.strftime('%Y-%m-%d')

            # Get options chain
            calls = stock.option_chain(date_str).calls
            puts = stock.option_chain(date_str).puts

            # Add option type column
            calls['optionType'] = 'call'
            puts['optionType'] = 'put'

            # Combine calls and puts
            options_chain = pd.concat([calls, puts])

            # Add metrics we need for pricing
            options_chain['expiryDate'] = date_str
            options_chain['daysToExpiry'] = (expiry_date - dt.datetime.now()).days

            return options_chain

        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return pd.DataFrame()

    def get_risk_free_rate(self, term: float) -> float:
        """
        Get approximate risk-free rate based on US Treasury yields

        Args:
            term: Term in years

        Returns:
            Approximate risk-free rate for the given term
        """
        logger.info(f"Fetching risk-free rate for term: {term} years")

        # For simplicity, we'll use fixed rates based on term
        # In a production system, you would fetch actual treasury yields
        if term <= 0.25:
            return 0.05  # 3-month rate
        elif term <= 0.5:
            return 0.052  # 6-month rate
        elif term <= 1:
            return 0.055  # 1-year rate
        elif term <= 2:
            return 0.058  # 2-year rate
        elif term <= 5:
            return 0.06  # 5-year rate
        elif term <= 10:
            return 0.062  # 10-year rate
        else:
            return 0.065  # 30-year rate


# ========================
# OPTION CONTRACT MODELS
# ========================

@dataclass
class OptionContract:
    """Class representing an option contract"""
    ticker: str
    strike: float
    expiry_date: dt.datetime
    option_type: str  # 'call' or 'put'
    market_price: Optional[float] = None
    implied_vol: Optional[float] = None

    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiry"""
        return (self.expiry_date - dt.datetime.now()).days

    @property
    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years"""
        return self.days_to_expiry / 365.0

    def __str__(self) -> str:
        return (f"{self.ticker} {self.option_type.upper()} {self.strike} "
                f"exp {self.expiry_date.strftime('%Y-%m-%d')}")


# ========================
# PRICING MODELS
# ========================

class PricingModel(ABC):
    """Abstract base class for all option pricing models"""

    @abstractmethod
    def calculate_price(self, option: OptionContract, spot: float,
                        risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> float:
        """Calculate theoretical price of an option"""
        pass

    @abstractmethod
    def calculate_greeks(self, option: OptionContract, spot: float,
                         risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> Dict[str, float]:
        """Calculate option Greeks"""
        pass

    def implied_volatility(self, option: OptionContract, spot: float,
                           risk_free_rate: float, dividend_yield: float = 0.0) -> float:
        """Calculate implied volatility from market price"""
        if option.market_price is None:
            raise ValueError("Market price is required to calculate implied volatility")

        def objective(vol):
            price = self.calculate_price(option, spot, risk_free_rate, vol, dividend_yield)
            return price - option.market_price

        # Try to solve for implied vol using optimization
        try:
            result = optimize.brentq(objective, 0.0001, 5.0)
            return result
        except Exception as e:
            logger.warning(f"Could not solve for implied volatility: {e}")
            return np.nan


class BlackScholesModel(PricingModel):
    """Black-Scholes option pricing model"""

    def calculate_price(self, option: OptionContract, spot: float,
                        risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> float:
        """
        Calculate Black-Scholes option price

        Args:
            option: Option contract details
            spot: Current spot price of the underlying
            risk_free_rate: Risk-free interest rate (annual, decimal)
            volatility: Volatility of the underlying (annual, decimal)
            dividend_yield: Continuous dividend yield (annual, decimal)

        Returns:
            Theoretical option price
        """
        K = option.strike
        T = option.time_to_expiry
        r = risk_free_rate
        q = dividend_yield
        sigma = volatility

        if T <= 0:
            # Option is expired
            if option.option_type.lower() == 'call':
                return max(0, spot - K)
            else:
                return max(0, K - spot)

        d1 = (np.log(spot / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option.option_type.lower() == 'call':
            price = spot * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - spot * np.exp(-q * T) * norm.cdf(-d1)

        return price

    def calculate_greeks(self, option: OptionContract, spot: float,
                         risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> Dict[str, float]:
        """Calculate option Greeks using Black-Scholes model"""
        K = option.strike
        T = option.time_to_expiry
        r = risk_free_rate
        q = dividend_yield
        sigma = volatility
        is_call = option.option_type.lower() == 'call'

        if T <= 0:
            # Return zeros for expired options
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0,
                'vega': 0.0, 'rho': 0.0
            }

        d1 = (np.log(spot / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate Greeks
        if is_call:
            delta = np.exp(-q * T) * norm.cdf(d1)
            theta = -((spot * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))) - \
                    (r * K * np.exp(-r * T) * norm.cdf(d2)) + \
                    (q * spot * np.exp(-q * T) * norm.cdf(d1))
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
            theta = -((spot * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))) + \
                    (r * K * np.exp(-r * T) * norm.cdf(-d2)) - \
                    (q * spot * np.exp(-q * T) * norm.cdf(-d1))

        gamma = (norm.pdf(d1) * np.exp(-q * T)) / (spot * sigma * np.sqrt(T))
        vega = spot * np.sqrt(T) * norm.pdf(d1) * np.exp(-q * T) / 100  # Divided by 100 for 1% change

        if is_call:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Divided by 100 for 1% change
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.0,  # Daily theta
            'vega': vega,
            'rho': rho
        }


class MonteCarloModel(PricingModel):
    """Monte Carlo simulation for option pricing"""

    def __init__(self, num_simulations: int = 10000, num_steps: int = 252):
        """
        Initialize Monte Carlo model

        Args:
            num_simulations: Number of price path simulations
            num_steps: Number of time steps in each simulation
        """
        self.num_simulations = num_simulations
        self.num_steps = num_steps

    def simulate_paths(self, spot: float, risk_free_rate: float, volatility: float,
                       time_to_expiry: float, dividend_yield: float = 0.0) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion

        Args:
            spot: Current spot price
            risk_free_rate: Risk-free rate (annual)
            volatility: Volatility (annual)
            time_to_expiry: Time to expiry in years
            dividend_yield: Continuous dividend yield (annual)

        Returns:
            Array of simulated prices at expiry
        """
        dt = time_to_expiry / self.num_steps
        nudt = (risk_free_rate - dividend_yield - 0.5 * volatility ** 2) * dt
        volsdt = volatility * np.sqrt(dt)

        # Initialize price paths
        paths = np.zeros((self.num_simulations, self.num_steps + 1))
        paths[:, 0] = spot

        # Generate random numbers for simulations
        Z = np.random.standard_normal((self.num_simulations, self.num_steps))

        # Simulate price paths
        for t in range(1, self.num_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(nudt + volsdt * Z[:, t - 1])

        return paths[:, -1]  # Return terminal prices

    def calculate_price(self, option: OptionContract, spot: float,
                        risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> float:
        """Calculate option price using Monte Carlo simulation"""
        # Simulate terminal stock prices
        terminal_prices = self.simulate_paths(
            spot, risk_free_rate, volatility, option.time_to_expiry, dividend_yield
        )

        # Calculate payoffs
        if option.option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - option.strike, 0)
        else:  # put
            payoffs = np.maximum(option.strike - terminal_prices, 0)

        # Calculate present value of expected payoff
        option_price = np.mean(payoffs) * np.exp(-risk_free_rate * option.time_to_expiry)

        return option_price

    def calculate_greeks(self, option: OptionContract, spot: float,
                         risk_free_rate: float, volatility: float, dividend_yield: float = 0.0) -> Dict[str, float]:
        """
        Calculate option Greeks using finite difference approximations

        Note: Monte Carlo Greeks are typically less stable than analytical solutions
        """
        # Calculate price
        price = self.calculate_price(option, spot, risk_free_rate, volatility, dividend_yield)

        # Small changes for finite differences
        h_spot = spot * 0.01  # 1% change in spot
        h_vol = volatility * 0.01  # 1% change in vol
        h_rate = 0.0001  # 1 basis point change in rate
        h_time = 1 / 365  # 1 day change

        # Delta: Δ = ∂V/∂S
        price_up_spot = self.calculate_price(option, spot + h_spot, risk_free_rate, volatility, dividend_yield)
        price_down_spot = self.calculate_price(option, spot - h_spot, risk_free_rate, volatility, dividend_yield)
        delta = (price_up_spot - price_down_spot) / (2 * h_spot)

        # Gamma: Γ = ∂²V/∂S²
        gamma = (price_up_spot - 2 * price + price_down_spot) / (h_spot ** 2)

        # Vega: ν = ∂V/∂σ
        price_up_vol = self.calculate_price(option, spot, risk_free_rate, volatility + h_vol, dividend_yield)
        vega = (price_up_vol - price) / h_vol * 0.01  # Scaling for 1% change

        # Theta: Θ = -∂V/∂t
        option_tomorrow = OptionContract(
            ticker=option.ticker,
            strike=option.strike,
            expiry_date=option.expiry_date,
            option_type=option.option_type,
            market_price=option.market_price,
            implied_vol=option.implied_vol
        )
        # Adjust time to expiry to be one day less
        option_tomorrow.expiry_date = option.expiry_date - dt.timedelta(days=1)
        price_tomorrow = self.calculate_price(option_tomorrow, spot, risk_free_rate, volatility, dividend_yield)
        theta = -(price_tomorrow - price) / h_time

        # Rho: ρ = ∂V/∂r
        price_up_rate = self.calculate_price(option, spot, risk_free_rate + h_rate, volatility, dividend_yield)
        rho = (price_up_rate - price) / h_rate * 0.01  # Scaling for 1% change

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365.0,  # Daily theta
            'vega': vega,
            'rho': rho
        }


# ========================
# OPTIONS PRICER
# ========================

class OptionsPricer:
    """Main options pricing engine"""

    def __init__(self, data_provider: DataProvider):
        """
        Initialize options pricer

        Args:
            data_provider: Data provider instance
        """
        self.data_provider = data_provider
        self.pricing_models = {
            'black_scholes': BlackScholesModel(),
            'monte_carlo': MonteCarloModel()
        }

    def get_market_data(self, ticker: str, lookback_days: int = 252) -> Dict:
        """
        Get market data required for option pricing

        Args:
            ticker: Ticker symbol
            lookback_days: Number of historical days to fetch for volatility calculation

        Returns:
            Dictionary with market data
        """
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days)

        # Get historical price data
        historical_data = self.data_provider.get_historical_prices(ticker, start_date, end_date)

        if historical_data.empty:
            raise ValueError(f"Could not get historical data for {ticker}")

        # Calculate historical volatility
        returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1)).dropna()
        hist_volatility = returns.std() * np.sqrt(252)  # Annualized

        # Get current price
        current_price = historical_data['Close'].iloc[-1]

        return {
            'ticker': ticker,
            'current_price': current_price,
            'historical_volatility': hist_volatility,
            'historical_data': historical_data
        }

    def get_options_data(self, ticker: str, expiry_date: Optional[dt.datetime] = None) -> pd.DataFrame:
        """
        Get options chain data

        Args:
            ticker: Ticker symbol
            expiry_date: Expiry date (optional)

        Returns:
            DataFrame with options data
        """
        return self.data_provider.get_options_chain(ticker, expiry_date)

    def create_option_contract(self, ticker: str, strike: float, expiry_date: dt.datetime,
                               option_type: str, market_price: Optional[float] = None) -> OptionContract:
        """
        Create an option contract object

        Args:
            ticker: Ticker symbol
            strike: Strike price
            expiry_date: Expiry date
            option_type: Option type ('call' or 'put')
            market_price: Market price (optional)

        Returns:
            OptionContract object
        """
        return OptionContract(
            ticker=ticker,
            strike=strike,
            expiry_date=expiry_date,
            option_type=option_type,
            market_price=market_price
        )

    def price_option(self, option: OptionContract, model_name: str = 'black_scholes',
                     spot: Optional[float] = None, risk_free_rate: Optional[float] = None,
                     volatility: Optional[float] = None, dividend_yield: float = 0.0) -> Dict:
        """
        Price an option using specified model

        Args:
            option: Option contract
            model_name: Pricing model to use
            spot: Spot price (optional, will fetch if not provided)
            risk_free_rate: Risk-free rate (optional, will fetch if not provided)
            volatility: Volatility (optional, will use historical if not provided)
            dividend_yield: Continuous dividend yield

        Returns:
            Dictionary with pricing results
        """
        if model_name not in self.pricing_models:
            raise ValueError(f"Unknown pricing model: {model_name}")

        model = self.pricing_models[model_name]

        # Get missing market data if needed
        if spot is None:
            market_data = self.get_market_data(option.ticker)
            spot = market_data['current_price']

        if risk_free_rate is None:
            risk_free_rate = self.data_provider.get_risk_free_rate(option.time_to_expiry)

        if volatility is None:
            # Try to use implied vol if market price is available
            if option.market_price is not None:
                try:
                    volatility = model.implied_volatility(option, spot, risk_free_rate, dividend_yield)
                except:
                    # Fall back to historical vol
                    market_data = self.get_market_data(option.ticker)
                    volatility = market_data['historical_volatility']
            else:
                # Use historical vol
                market_data = self.get_market_data(option.ticker)
                volatility = market_data['historical_volatility']

        # Price the option
        price = model.calculate_price(option, spot, risk_free_rate, volatility, dividend_yield)
        greeks = model.calculate_greeks(option, spot, risk_free_rate, volatility, dividend_yield)

        # If market price is available, calculate the difference
        price_diff = None
        if option.market_price is not None:
            price_diff = option.market_price - price

        return {
            'model': model_name,
            'price': price,
            'greeks': greeks,
            'inputs': {
                'spot': spot,
                'strike': option.strike,
                'time_to_expiry': option.time_to_expiry,
                'risk_free_rate': risk_free_rate,
                'volatility': volatility,
                'dividend_yield': dividend_yield,
                'option_type': option.option_type
            },
            'market_price': option.market_price,
            'price_diff': price_diff
        }

    def compare_models(self, option: OptionContract, spot: Optional[float] = None,
                       risk_free_rate: Optional[float] = None, volatility: Optional[float] = None,
                       dividend_yield: float = 0.0) -> Dict:
        """
        Compare pricing results between different models

        Args:
            option: Option contract
            spot: Spot price (optional)
            risk_free_rate: Risk-free rate (optional)
            volatility: Volatility (optional)
            dividend_yield: Continuous dividend yield

        Returns:
            Dictionary with comparison results
        """
        results = {}

        for model_name in self.pricing_models:
            results[model_name] = self.price_option(
                option, model_name, spot, risk_free_rate, volatility, dividend_yield
            )

        return results

    def plot_volatility_surface(self, ticker: str, expiry_dates: List[dt.datetime] = None) -> plt.Figure:
        """
        Plot implied volatility surface

        Args:
            ticker: Ticker symbol
            expiry_dates: List of expiry dates to include

        Returns:
            Matplotlib figure with volatility surface
        """
        # Get market data
        market_data = self.get_market_data(ticker)
        spot = market_data['current_price']

        # If no expiry dates specified, get all available
        if expiry_dates is None:
            stock = yf.Ticker(ticker)
            expiry_dates = [dt.datetime.strptime(date, '%Y-%m-%d') for date in stock.options]

        # Get options data for all expiries
        vol_data = []
        model = self.pricing_models['black_scholes']

        for expiry in expiry_dates:
            options_chain = self.get_options_data(ticker, expiry)
            if options_chain.empty:
                continue

            time_to_expiry = (expiry - dt.datetime.now()).days / 365.0
            rf_rate = self.data_provider.get_risk_free_rate(time_to_expiry)

            # Process calls and puts separately
            for opt_type in ['call', 'put']:
                options = options_chain[options_chain['optionType'] == opt_type]

                for _, row in options.iterrows():
                    option = OptionContract(
                        ticker=ticker,
                        strike=row['strike'],
                        expiry_date=expiry,
                        option_type=opt_type,
                        market_price=row['lastPrice']
                    )

                    try:
                        impl_vol = model.implied_volatility(option, spot, rf_rate)
                        if not np.isnan(impl_vol) and impl_vol > 0 and impl_vol < 2.0:  # Filter unrealistic values
                            vol_data.append({
                                'strike': row['strike'],
                                'moneyness': row['strike'] / spot,
                                'time_to_expiry': time_to_expiry,
                                'implied_vol': impl_vol,
                                'option_type': opt_type
                            })
                    except:
                        pass

        # Create DataFrame from collected data
        vol_df = pd.DataFrame(vol_data)

        if vol_df.empty:
            logger.warning("No valid implied volatility data")
            return None

        # Plot volatility surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot calls and puts with different colors
        colors = {'call': 'blue', 'put': 'red'}

        for opt_type, group in vol_df.groupby('option_type'):
            ax.scatter(
                group['moneyness'],
                group['time_to_expiry'],
                group['implied_vol'],
                color=colors[opt_type],
                label=f"{opt_type.capitalize()}s"
            )

        ax.set_xlabel('Moneyness (Strike/Spot)')
        ax.set_ylabel('Time to Expiry (Years)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'Implied Volatility Surface for {ticker}')
        ax.legend()

        return fig


# ========================
# USAGE EXAMPLE
# ========================

def run_example():
    """Example usage of the options pricing framework"""

    # Create data provider
    provider = YahooFinanceProvider()

    # Create options pricer
    pricer = OptionsPricer(provider)

    # Get market data for SPY
    market_data = pricer.get_market_data('SPY')
    print(f"Current SPY price: ${market_data['current_price']:.2f}")
    print(f"Historical volatility: {market_data['historical_volatility']:.2%}")

    # Get options chain data
    options_chain = pricer.get_options_data('SPY')
    print(f"Retrieved {len(options_chain)} option contracts")

    # Select a call option near the money
    spot = market_data['current_price']
    near_money_calls = options_chain[
        (options_chain['optionType'] == 'call') &
        (options_chain['strike'] > spot * 0.95) &
        (options_chain['strike'] < spot * 1.05)
        ]

    if not near_money_calls.empty:
        selected_option = near_money_calls.iloc[0]
        print(f"\nSelected option: SPY {selected_option['strike']} Call expiring on {selected_option['expiryDate']}")
        print(f"Market price: ${selected_option['lastPrice']:.2f}")

        # Create option contract
        expiry_date = dt.datetime.strptime(selected_option['expiryDate'], '%Y-%m-%d')
        option = pricer.create_option_contract(
            ticker='SPY',
            strike=selected_option['strike'],
            expiry_date=expiry_date,
            option_type='call',
            market_price=selected_option['lastPrice']
        )

        # Price with Black-Scholes
        bs_result = pricer.price_option(option, 'black_scholes')
        print("\nBlack-Scholes pricing:")
        print(f"Theoretical price: ${bs_result['price']:.2f}")
        print(f"Implied volatility: {bs_result['inputs']['volatility']:.2%}")
        print("Greeks:")
        for greek, value in bs_result['greeks'].items():
            print(f"  {greek.capitalize()}: {value:.4f}")

        # Price with Monte Carlo
        mc_result = pricer.price_option(option, 'monte_carlo')
        print("\nMonte Carlo pricing:")
        print(f"Theoretical price: ${mc_result['price']:.2f}")
        print(f"Difference from BS: ${mc_result['price'] - bs_result['price']:.4f}")

        # Compare with market price
        print("\nMispricing analysis:")
        print(f"Market price: ${option.market_price:.2f}")
        print(f"BS model price: ${bs_result['price']:.2f}")
        print(f"Difference: ${bs_result['price_diff']:.2f} ({bs_result['price_diff'] / option.market_price:.2%})")

        # Plot implied volatility smile
        try:
            fig = pricer.plot_volatility_surface('SPY')
            plt.show()
        except Exception as e:
            print(f"Could not plot volatility surface: {e}")
    else:
        print("No suitable options found")

    if __name__ == "__main__":
        run_example()