"""
Energy Derivatives Pricing Framework
-----------------------------------
Core module for pricing various energy derivatives with both analytical
and Monte Carlo methods, with real market data integration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm
from scipy import optimize
import requests
import json
from typing import Dict, List, Tuple, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataProvider:
    """Class to retrieve and process market data from external APIs"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.energydata.com/v1/"  # Example API endpoint

    def get_forward_curve(self, commodity: str, delivery_location: str) -> pd.DataFrame:
        """
        Retrieve forward curve data for a specific commodity and location

        Args:
            commodity: The commodity code (e.g., 'NG' for natural gas, 'PW' for power)
            delivery_location: Location code (e.g., 'HH' for Henry Hub, 'PJM' for PJM)

        Returns:
            DataFrame with forward curve data
        """
        try:
            endpoint = f"forwards/{commodity}/{delivery_location}"
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers={"X-API-KEY": self.api_key}
            )
            response.raise_for_status()

            data = response.json()

            # Process the data into a DataFrame
            curve_data = pd.DataFrame(data['forward_prices'])
            curve_data['delivery_date'] = pd.to_datetime(curve_data['delivery_date'])
            curve_data.set_index('delivery_date', inplace=True)

            logger.info(f"Successfully retrieved forward curve for {commodity} at {delivery_location}")
            return curve_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve forward curve: {str(e)}")
            # In a real scenario, might implement fallback data or raise a custom exception
            # For demo, return a synthetic curve
            return self._get_synthetic_forward_curve(commodity, delivery_location)

    def get_volatility_surface(self, commodity: str, delivery_location: str) -> pd.DataFrame:
        """
        Retrieve volatility surface data for option pricing

        Args:
            commodity: The commodity code
            delivery_location: Location code

        Returns:
            DataFrame with volatility surface data
        """
        try:
            endpoint = f"volatility/{commodity}/{delivery_location}"
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers={"X-API-KEY": self.api_key}
            )
            response.raise_for_status()

            data = response.json()

            # Process the data into a DataFrame
            vol_data = pd.DataFrame(data['volatility_surface'])
            vol_data['expiry_date'] = pd.to_datetime(vol_data['expiry_date'])

            logger.info(f"Successfully retrieved volatility surface for {commodity} at {delivery_location}")
            return vol_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve volatility surface: {str(e)}")
            return self._get_synthetic_volatility_surface(commodity, delivery_location)

    def _get_synthetic_forward_curve(self, commodity: str, delivery_location: str) -> pd.DataFrame:
        """Generate synthetic forward curve for demo/testing"""
        logger.warning("Using synthetic forward curve data")

        today = dt.date.today()
        dates = [today + dt.timedelta(days=30*i) for i in range(24)]

        if commodity == 'NG':  # Natural Gas
            # Create seasonal pattern for natural gas
            base_price = 4.5  # $/MMBtu
            prices = [base_price * (1 + 0.2 * np.sin(2 * np.pi * i / 12)) for i in range(24)]
        elif commodity == 'PW':  # Power
            # Create seasonal pattern for power
            base_price = 50.0  # $/MWh
            prices = [base_price * (1 + 0.15 * np.sin(2 * np.pi * i / 12) + 0.05 * np.random.randn()) for i in range(24)]
        else:
            base_price = 50.0
            prices = [base_price * (1 + 0.01 * i + 0.03 * np.random.randn()) for i in range(24)]

        df = pd.DataFrame({
            'delivery_date': dates,
            'forward_price': prices
        })
        df.set_index('delivery_date', inplace=True)
        return df

    def _get_synthetic_volatility_surface(self, commodity: str, delivery_location: str) -> pd.DataFrame:
        """Generate synthetic volatility surface for demo/testing"""
        logger.warning("Using synthetic volatility surface data")

        today = dt.date.today()
        expiries = [30, 60, 90, 180, 270, 365]
        strikes = [0.8, 0.9, 1.0, 1.1, 1.2]

        data = []
        for expiry_days in expiries:
            expiry_date = today + dt.timedelta(days=expiry_days)
            for strike_mult in strikes:
                # Volatility smile effect
                vol = 0.3 + 0.1 * (strike_mult - 1.0)**2
                # Term structure effect - longer dated options have lower vol
                vol *= max(0.7, 1.0 - 0.0005 * expiry_days)

                data.append({
                    'expiry_date': expiry_date,
                    'strike_multiplier': strike_mult,
                    'volatility': vol
                })

        return pd.DataFrame(data)


class EnergyOption:
    """Base class for energy options pricing"""

    def __init__(
        self,
        option_type: str,  # 'call' or 'put'
        strike: float,
        expiry_date: dt.date,
        underlying: str,  # e.g., 'NG', 'PW'
        location: str,    # e.g., 'HH', 'PJM'
        contract_size: float = 1.0
    ):
        self.option_type = option_type.lower()
        self.strike = strike
        self.expiry_date = expiry_date
        self.underlying = underlying
        self.location = location
        self.contract_size = contract_size

        if self.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")

    def price_black_scholes(self, forward_price: float, volatility: float,
                          risk_free_rate: float, time_to_expiry: float) -> float:
        """
        Price the option using Black-Scholes model (for European options)

        Args:
            forward_price: Current forward price for the underlying
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate (annualized)
            time_to_expiry: Time to expiry in years

        Returns:
            Option price
        """
        # Calculate d1 and d2
        if volatility <= 0 or time_to_expiry <= 0:
            return max(0, (forward_price - self.strike) if self.option_type == 'call' else (self.strike - forward_price))

        d1 = (np.log(forward_price / self.strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Calculate option price
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)

        if self.option_type == 'call':
            price = discount_factor * (forward_price * norm.cdf(d1) - self.strike * norm.cdf(d2))
        else:  # put
            price = discount_factor * (self.strike * norm.cdf(-d2) - forward_price * norm.cdf(-d1))

        return price * self.contract_size

    def price_monte_carlo(self, forward_price: float, volatility: float, risk_free_rate: float,
                         time_to_expiry: float, n_simulations: int = 10000,
                         mean_reversion_speed: Optional[float] = None,
                         jump_intensity: Optional[float] = None,
                         jump_size_mean: Optional[float] = None,
                         jump_size_std: Optional[float] = None) -> Tuple[float, float]:
        """
        Price the option using Monte Carlo simulation with optional mean-reversion and jumps

        Args:
            forward_price: Current forward price for the underlying
            volatility: Implied volatility (annualized)
            risk_free_rate: Risk-free interest rate (annualized)
            time_to_expiry: Time to expiry in years
            n_simulations: Number of price path simulations
            mean_reversion_speed: Speed of mean reversion (if None, GBM is used)
            jump_intensity: Expected number of jumps per year (if None, no jumps)
            jump_size_mean: Mean of jump size (as fraction of price)
            jump_size_std: Standard deviation of jump size

        Returns:
            Tuple of (option_price, standard_error)
        """
        dt = 1/252  # Daily steps
        n_steps = int(time_to_expiry / dt)

        # Initialize price paths
        price_paths = np.zeros((n_simulations, n_steps + 1))
        price_paths[:, 0] = forward_price

        # Generate price paths
        for t in range(1, n_steps + 1):
            # Random normal innovations
            z = np.random.normal(0, 1, n_simulations)

            # Diffusion component
            if mean_reversion_speed is None:
                # Standard GBM
                drift = (risk_free_rate - 0.5 * volatility**2) * dt
                diffusion = volatility * np.sqrt(dt) * z
                price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion)
            else:
                # Mean-reverting process (Ornstein-Uhlenbeck)
                long_term_mean = forward_price  # Simplification: assuming forward is the mean
                dr = mean_reversion_speed * (long_term_mean - price_paths[:, t-1]) * dt + \
                     volatility * price_paths[:, t-1] * np.sqrt(dt) * z
                price_paths[:, t] = price_paths[:, t-1] + dr

            # Add jumps if specified
            if jump_intensity is not None:
                # Generate jump occurrences (Poisson process)
                jump_occurs = np.random.poisson(jump_intensity * dt, n_simulations) > 0

                # Generate jump sizes where jumps occur
                if np.any(jump_occurs):
                    jump_sizes = np.random.normal(jump_size_mean, jump_size_std, n_simulations)
                    jump_sizes[~jump_occurs] = 0
                    price_paths[:, t] = price_paths[:, t] * (1 + jump_sizes)

        # Calculate option payoffs at expiry
        if self.option_type == 'call':
            payoffs = np.maximum(0, price_paths[:, -1] - self.strike)
        else:  # put
            payoffs = np.maximum(0, self.strike - price_paths[:, -1])

        # Discount payoffs to present value
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        present_values = payoffs * discount_factor * self.contract_size

        # Calculate option price and standard error
        option_price = np.mean(present_values)
        std_error = np.std(present_values) / np.sqrt(n_simulations)

        return option_price, std_error


class NaturalGasOption(EnergyOption):
    """Class for natural gas options with specific characteristics"""

    def __init__(
        self,
        option_type: str,
        strike: float,
        expiry_date: dt.date,
        location: str = 'HH',  # Default to Henry Hub
        contract_size: float = 10000,  # Default contract size in MMBtu
        is_physical: bool = False  # Physical delivery vs cash settlement
    ):
        super().__init__(option_type, strike, expiry_date, 'NG', location, contract_size)
        self.is_physical = is_physical

    def price_with_seasonal_volatility(self, market_data_provider: MarketDataProvider,
                                      risk_free_rate: float, pricing_date: dt.date = None) -> Dict:
        """
        Price the option with seasonal volatility adjustment

        Args:
            market_data_provider: Provider for market data
            risk_free_rate: Risk-free rate
            pricing_date: Date for pricing (defaults to today)

        Returns:
            Dictionary with pricing results
        """
        if pricing_date is None:
            pricing_date = dt.date.today()

        # Get market data
        forward_curve = market_data_provider.get_forward_curve('NG', self.location)
        vol_surface = market_data_provider.get_volatility_surface('NG', self.location)

        # Find appropriate forward price for delivery period
        forward_price = self._get_forward_price(forward_curve, self.expiry_date)

        # Calculate time to expiry
        time_to_expiry = (self.expiry_date - pricing_date).days / 365.0

        # Find appropriate volatility from surface
        volatility = self._get_volatility(vol_surface, time_to_expiry, self.strike / forward_price)

        # Apply seasonal adjustment to volatility
        # Natural gas typically has higher volatility in winter months
        month = self.expiry_date.month
        if month in [12, 1, 2]:  # Winter months
            seasonal_factor = 1.2
        elif month in [6, 7, 8]:  # Summer months
            seasonal_factor = 0.9
        else:
            seasonal_factor = 1.0

        adjusted_volatility = volatility * seasonal_factor

        # Price with Black-Scholes
        bs_price = self.price_black_scholes(
            forward_price, adjusted_volatility, risk_free_rate, time_to_expiry
        )

        # Price with Monte Carlo (with mean reversion)
        mc_price, mc_error = self.price_monte_carlo(
            forward_price, adjusted_volatility, risk_free_rate, time_to_expiry,
            n_simulations=10000,
            mean_reversion_speed=2.5,  # Natural gas typically shows mean reversion
            jump_intensity=3.0,        # Jumps due to weather events
            jump_size_mean=0.03,       # Small positive bias in jumps
            jump_size_std=0.08         # Jump size variation
        )

        return {
            'pricing_date': pricing_date,
            'forward_price': forward_price,
            'implied_volatility': volatility,
            'adjusted_volatility': adjusted_volatility,
            'black_scholes_price': bs_price,
            'monte_carlo_price': mc_price,
            'monte_carlo_error': mc_error,
            'time_to_expiry': time_to_expiry
        }

    def _get_forward_price(self, forward_curve: pd.DataFrame, delivery_date: dt.date) -> float:
        """Extract appropriate forward price for the delivery date"""
        # Find closest date in forward curve
        if delivery_date in forward_curve.index:
            return forward_curve.loc[delivery_date, 'forward_price']
        else:
            # Find closest date - in a real system this would use interpolation
            closest_date = min(forward_curve.index, key=lambda x: abs((x - pd.Timestamp(delivery_date)).days))
            return forward_curve.loc[closest_date, 'forward_price']

    def _get_volatility(self, vol_surface: pd.DataFrame, time_to_expiry: float,
                       strike_ratio: float) -> float:
        """Extract appropriate volatility from surface with interpolation"""
        # In a real system, this would use proper 2D interpolation
        # For this demo, we'll use a simple approach

        # Convert time_to_expiry to days for matching
        tte_days = int(time_to_expiry * 365)

        # Find closest expiry
        closest_expiry = min(vol_surface['expiry_date'].unique(),
                           key=lambda x: abs((x - pd.Timestamp.today()).days - tte_days))

        # Filter for this expiry
        expiry_slice = vol_surface[vol_surface['expiry_date'] == closest_expiry]

        # Find closest strike ratio
        closest_strike = min(expiry_slice['strike_multiplier'],
                           key=lambda x: abs(x - strike_ratio))

        # Get volatility
        vol = expiry_slice[expiry_slice['strike_multiplier'] == closest_strike]['volatility'].values[0]

        return vol


class PowerOption(EnergyOption):
    """Class for power options with specific characteristics"""

    def __init__(
        self,
        option_type: str,
        strike: float,
        expiry_date: dt.date,
        location: str = 'PJM',  # Default to PJM
        peak_type: str = 'baseload',  # 'baseload', 'peak', 'off-peak'
        contract_size: float = 40,  # Default contract size in MWh (e.g., 40 MWh/day for a month)
    ):
        super().__init__(option_type, strike, expiry_date, 'PW', location, contract_size)
        self.peak_type = peak_type

        if self.peak_type not in ['baseload', 'peak', 'off-peak']:
            raise ValueError("Peak type must be 'baseload', 'peak', or 'off-peak'")

    def price_with_spikes(self, market_data_provider: MarketDataProvider,
                        risk_free_rate: float, pricing_date: dt.date = None) -> Dict:
        """
        Price power option with consideration for price spikes

        Args:
            market_data_provider: Provider for market data
            risk_free_rate: Risk-free rate
            pricing_date: Date for pricing (defaults to today)

        Returns:
            Dictionary with pricing results
        """
        if pricing_date is None:
            pricing_date = dt.date.today()

        # Get market data
        forward_curve = market_data_provider.get_forward_curve('PW', self.location)
        vol_surface = market_data_provider.get_volatility_surface('PW', self.location)

        # Apply adjustment for peak type
        if self.peak_type == 'peak':
            peak_factor = 1.3  # Peak power typically trades at premium
        elif self.peak_type == 'off-peak':
            peak_factor = 0.7  # Off-peak typically trades at discount
        else:  # baseload
            peak_factor = 1.0

        # Find appropriate forward price for delivery period (adjusted for peak type)
        forward_price = self._get_forward_price(forward_curve, self.expiry_date) * peak_factor

        # Calculate time to expiry
        time_to_expiry = (self.expiry_date - pricing_date).days / 365.0

        # Find appropriate volatility from surface
        volatility = self._get_volatility(vol_surface, time_to_expiry, self.strike / forward_price)

        # Adjust volatility for peak type
        if self.peak_type == 'peak':
            vol_factor = 1.2  # Peak hours typically more volatile
        elif self.peak_type == 'off-peak':
            vol_factor = 0.8  # Off-peak typically less volatile
        else:
            vol_factor = 1.0

        adjusted_volatility = volatility * vol_factor

        # Price with Black-Scholes (as baseline)
        bs_price = self.price_black_scholes(
            forward_price, adjusted_volatility, risk_free_rate, time_to_expiry
        )

        # For power markets, jump-diffusion is particularly important due to price spikes
        # Set jump parameters based on peak type
        if self.peak_type == 'peak':
            jump_intensity = 10.0  # More frequent jumps during peak hours
            jump_size_mean = 0.10  # Larger positive jumps
            jump_size_std = 0.20
        elif self.peak_type == 'off-peak':
            jump_intensity = 3.0   # Fewer jumps in off-peak
            jump_size_mean = 0.05
            jump_size_std = 0.10
        else:  # baseload
            jump_intensity = 5.0
            jump_size_mean = 0.08
            jump_size_std = 0.15

        # Price with Monte Carlo (with jumps)
        mc_price, mc_error = self.price_monte_carlo(
            forward_price, adjusted_volatility, risk_free_rate, time_to_expiry,
            n_simulations=10000,
            mean_reversion_speed=4.0,    # Power prices mean-revert quickly
            jump_intensity=jump_intensity,
            jump_size_mean=jump_size_mean,
            jump_size_std=jump_size_std
        )

        return {
            'pricing_date': pricing_date,
            'forward_price': forward_price,
            'peak_type': self.peak_type,
            'implied_volatility': volatility,
            'adjusted_volatility': adjusted_volatility,
            'black_scholes_price': bs_price,
            'monte_carlo_price': mc_price,
            'monte_carlo_error': mc_error,
            'time_to_expiry': time_to_expiry
        }

    def _get_forward_price(self, forward_curve: pd.DataFrame, delivery_date: dt.date) -> float:
        """Extract appropriate forward price for the delivery date"""
        # Same implementation as NaturalGasOption
        if delivery_date in forward_curve.index:
            return forward_curve.loc[delivery_date, 'forward_price']
        else:
            closest_date = min(forward_curve.index, key=lambda x: abs((x - pd.Timestamp(delivery_date)).days))
            return forward_curve.loc[closest_date, 'forward_price']

    def _get_volatility(self, vol_surface: pd.DataFrame, time_to_expiry: float,
                       strike_ratio: float) -> float:
        """Extract appropriate volatility from surface with interpolation"""
        # Same implementation as NaturalGasOption
        tte_days = int(time_to_expiry * 365)

        closest_expiry = min(vol_surface['expiry_date'].unique(),
                           key=lambda x: abs((x - pd.Timestamp.today()).days - tte_days))

        expiry_slice = vol_surface[vol_surface['expiry_date'] == closest_expiry]

        closest_strike = min(expiry_slice['strike_multiplier'],
                           key=lambda x: abs(x - strike_ratio))

        vol = expiry_slice[expiry_slice['strike_multiplier'] == closest_strike]['volatility'].values[0]

        return vol


# Example usage
if __name__ == "__main__":
    # Initialize market data provider
    api_key = "your_api_key"  # In a real project, use environment variables for this
    market_data = MarketDataProvider(api_key)

    # Set current date and expiry date
    current_date = dt.date.today()
    expiry_date = current_date + dt.timedelta(days=90)  # 3-month option

    # Create a natural gas call option
    ng_option = NaturalGasOption(
        option_type='call',
        strike=4.50,  # Strike price in $/MMBtu
        expiry_date=expiry_date,
        location='HH',  # Henry Hub
        contract_size=10000  # Standard contract size
    )

    # Price the natural gas option
    ng_price_results = ng_option.price_with_seasonal_volatility(
        market_data_provider=market_data,
        risk_free_rate=0.04,  # 4% risk-free rate
        pricing_date=current_date
    )

    # Print the results
    print("\nNatural Gas Option Pricing Results:")
    for key, value in ng_price_results.items():
        print(f"{key}: {value}")

    # Create a power call option
    power_option = PowerOption(
        option_type='call',
        strike=55.0,  # Strike price in $/MWh
        expiry_date=expiry_date,
        location='PJM',
        peak_type='peak',
        contract_size=40  # 40 MWh per day
    )

    # Price the power option
    power_price_results = power_option.price_with_spikes(
        market_data_provider=market_data,
        risk_free_rate=0.04,  # 4% risk-free rate
        pricing_date=current_date
    )

    # Print the results
    print("\nPower Option Pricing Results:")
    for key, value in power_price_results.items():
        print(f"{key}: {value}")

    # Create example plot showing price paths from Monte Carlo simulation
    def plot_price_paths(forward_price, volatility, risk_free_rate, time_to_expiry,
                      mean_reversion=None, jump_intensity=None):
        """Generate and plot example price paths"""
        np.random.seed(42)  # For reproducibility

        n_paths = 20
        n_steps = 252
        dt = time_to_expiry / n_steps

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = forward_price

        for t in range(1, n_steps + 1):
            z = np.random.normal(0, 1, n_paths)

            if mean_reversion is None:
                # GBM
                drift = (risk_free_rate - 0.5 * volatility**2) * dt
                diffusion = volatility * np.sqrt(dt) * z
                paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)
            else:
                # Mean-reverting
                dr = mean_reversion * (forward_price - paths[:, t-1]) * dt + \
                     volatility * paths[:, t-1] * np.sqrt(dt) * z
                paths[:, t] = paths[:, t-1] + dr

            # Add jumps if specified
            if jump_intensity is not None:
                jump_occurs = np.random.poisson(jump_intensity * dt, n_paths) > 0
                if np.any(jump_occurs):
                    jump_sizes = np.random.normal(0.08, 0.15, n_paths)
                    jump_sizes[~jump_occurs] = 0
                    paths[:, t] = paths[:, t] * (1 + jump_sizes)

        # Plot
        plt.figure(figsize=(10, 6))
        time_points = np.linspace(0, time_to_expiry, n_steps + 1)
        for i in range(n_paths):
            plt.plot(time_points, paths[i, :])

        plt.xlabel('Time to Expiry (years)')
        plt.ylabel('Price')
        if mean_reversion is None and jump_intensity is None:
            plt.title('GBM Price Paths')
        elif mean_reversion is not None and jump_intensity is None:
            plt.title('Mean-Reverting Price Paths')
        elif jump_intensity is not None:
            plt.title('Jump-Diffusion Price Paths')

        plt.grid(True)
        plt.savefig(f"price_paths_{'jump' if jump_intensity else 'mr' if mean_reversion else 'gbm'}.png")
        plt.close()

    # Generate sample plots
    print("\nGenerating sample price path plots...")
    plot_price_paths(forward_price=4.5, volatility=0.3, risk_free_rate=0.04, time_to_expiry=1.0)
    plot_price_paths(forward_price=4.5, volatility=0.3, risk_free_rate=0.04, time_to_expiry=1.0, mean_reversion=2.5)
    plot_price_paths(forward_price=50.0, volatility=0.4, risk_free_rate=0.04, time_to_expiry=1.0,
                  mean_reversion=4.0, jump_intensity=5.0)
    print("Sample plots generated.")