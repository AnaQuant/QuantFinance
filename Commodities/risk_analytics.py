"""
Risk Analytics for Energy Derivatives
------------------------------------
Complement to the pricing framework with risk metrics calculation and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from typing import Dict, List, Tuple, Union, Optional
import logging
from scipy.stats import norm
import seaborn as sns

# Import our pricing module
# In a real project, this would be properly packaged
# from energy_derivatives_pricing import EnergyOption, MarketDataProvider, NaturalGasOption, PowerOption

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnergyPortfolio:
    """Class to manage a portfolio of energy derivatives and calculate risk metrics"""

    def __init__(self, name: str):
        self.name = name
        self.positions = []
        self.market_data = None
        self.risk_free_rate = 0.04  # Default risk-free rate

    def set_market_data_provider(self, market_data_provider: 'MarketDataProvider'):
        """Set the market data provider for the portfolio"""
        self.market_data = market_data_provider

    def add_position(self, instrument: Union['EnergyOption', 'EnergyForward'], quantity: float):
        """
        Add a position to the portfolio

        Args:
            instrument: The derivative instrument
            quantity: The quantity (+ve for long, -ve for short)
        """
        self.positions.append({
            'instrument': instrument,
            'quantity': quantity
        })
        logger.info(f"Added position: {instrument.__class__.__name__}, quantity: {quantity}")

    def calculate_portfolio_value(self, valuation_date: dt.date = None) -> float:
        """Calculate the total value of the portfolio"""
        if valuation_date is None:
            valuation_date = dt.date.today()

        if self.market_data is None:
            raise ValueError("Market data provider not set")

        total_value = 0.0

        for position in self.positions:
            instrument = position['instrument']
            quantity = position['quantity']

            if isinstance(instrument, NaturalGasOption):
                price_results = instrument.price_with_seasonal_volatility(
                    self.market_data, self.risk_free_rate, valuation_date
                )
                value = price_results['monte_carlo_price']

            elif isinstance(instrument, PowerOption):
                price_results = instrument.price_with_spikes(
                    self.market_data, self.risk_free_rate, valuation_date
                )
                value = price_results['monte_carlo_price']

            else:
                # Generic pricing for other instrument types
                # In a real system, each instrument type would have specific pricing method
                value = instrument.price(self.market_data, self.risk_free_rate, valuation_date)

            total_value += value * quantity

        return total_value

    def calculate_var(self, confidence_level: float = 0.95,
                      horizon_days: int = 1, n_simulations: int = 10000) -> Dict:
        """
        Calculate Value-at-Risk (VaR) for the portfolio

        Args:
            confidence_level: Confidence level for VaR (e.g., 0.95 for 95% VaR)
            horizon_days: Risk horizon in days
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Dictionary with VaR results
        """
        if self.market_data is None:
            raise ValueError("Market data provider not set")

        if not self.positions:
            return {'var': 0.0, 'cvar': 0.0}

        # Get current portfolio value as baseline
        current_value = self.calculate_portfolio_value()

        # Simulate market factor changes
        horizon_years = horizon_days / 365.0

        # In a real system, this would use a proper factor model with correlations
        # For demo purposes, we'll use a simplified approach
        simulated_values = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Simulate market factor changes
            # - Generate correlated shocks to commodity prices and volatilities
            # - Apply these shocks to each position
            # - Sum up the new position values

            # For demo, we'll just add a simple random shock to the portfolio value
            # In a real system, this would properly model each risk factor

            # Standard deviation of portfolio based on weighted instrument volatilities (simplified)
            portfolio_volatility = 0.02  # Example - in reality this would be calculated

            # Random shock to portfolio value
            shock = np.random.normal(0, portfolio_volatility * np.sqrt(horizon_years))
            simulated_values[i] = current_value * (1 + shock)

        # Calculate P&L distribution
        pnl = simulated_values - current_value

        # Calculate VaR
        var_index = int((1 - confidence_level) * n_simulations)
        sorted_pnl = np.sort(pnl)
        var = -sorted_pnl[var_index]  # VaR is positive for a loss

        # Calculate CVaR (Expected Shortfall)
        cvar = -np.mean(sorted_pnl[:var_index])

        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'horizon_days': horizon_days
        }

    def calculate_sensitivities(self) -> pd.DataFrame:
        """
        Calculate first-order sensitivities (Greeks) for all positions

        Returns:
            DataFrame with sensitivities
        """
        if self.market_data is None:
            raise ValueError("Market data provider not set")

        results = []

        for position in self.positions:
            instrument = position['instrument']
            quantity = position['quantity']

            # Extract base instrument info
            instrument_type = instrument.__class__.__name__
            underlying = getattr(instrument, 'underlying', 'Unknown')
            location = getattr(instrument, 'location', 'Unknown')

            # Base case pricing
            if isinstance(instrument, NaturalGasOption):
                base_results = instrument.price_with_seasonal_volatility(
                    self.market_data, self.risk_free_rate
                )
                forward_price = base_results['forward_price']
                volatility = base_results['adjusted_volatility']
                base_price = base_results['monte_carlo_price']

            elif isinstance(instrument, PowerOption):
                base_results = instrument.price_with_spikes(
                    self.market_data, self.risk_free_rate
                )
                forward_price = base_results['forward_price']
                volatility = base_results['adjusted_volatility']
                base_price = base_results['monte_carlo_price']

            else:
                # Generic approach for other instruments
                forward_price = 50.0  # Placeholder
                volatility = 0.3  # Placeholder
                base_price = instrument.price(self.market_data, self.risk_free_rate)

            # Calculate Delta (∂Price/∂S)
            bump_size = forward_price * 0.01  # 1% bump

            # Up bump
            if hasattr(instrument, 'price_with_seasonal_volatility'):
                # Need to modify the pricing to accept a custom forward price
                # This is simplified - in a real system we'd have a proper repricing
                up_price = base_price * (1 + 0.01)
            else:
                up_price = base_price * (1 + 0.01)

            # Down bump
            down_price = base_price * (1 - 0.01)

            # Central difference approximation
            delta = (up_price - down_price) / (2 * bump_size)
            delta *= quantity  # Scale by position size

            # Calculate Gamma (∂²Price/∂S²)
            gamma = (up_price - 2 * base_price + down_price) / (bump_size ** 2)
            gamma *= quantity

            # Calculate Vega (∂Price/∂σ)
            vol_bump = 0.01  # 100 basis points

            # Up vol bump - simplified approach
            up_vol_price = base_price * (1 + 0.05)

            # Down vol bump
            down_vol_price = base_price * (1 - 0.05)

            vega = (up_vol_price - down_vol_price) / (2 * vol_bump)
            vega *= quantity

            # Add results
            results.append({
                'instrument_type': instrument_type,
                'underlying': underlying,
                'location': location,
                'quantity': quantity,
                'base_price': base_price,
                'delta': delta,
                'gamma': gamma,
                'vega': vega
            })

        return pd.DataFrame(results)

    def run_stress_test(self, stress_scenarios: List[Dict]) -> pd.DataFrame:
        """
        Run stress tests on the portfolio

        Args:
            stress_scenarios: List of scenarios, each a dict with factor shocks

        Returns:
            DataFrame with stress test results
        """
        if self.market_data is None:
            raise ValueError("Market data provider not set")

        baseline_value = self.calculate_portfolio_value()
        results = []

        for scenario in stress_scenarios:
            scenario_name = scenario.get('name', 'Unnamed Scenario')
            price_shock = scenario.get('price_shock', 0.0)
            vol_shock = scenario.get('vol_shock', 0.0)

            # In a real system, would apply the shocks to market data and reprice
            # For demo, we'll use a simplified approach based on sensitivities

            # Get sensitivities
            sensitivities = self.calculate_sensitivities()

            # Calculate impact from price shock
            delta_impact = sensitivities['delta'].sum() * price_shock
            gamma_impact = 0.5 * sensitivities['gamma'].sum() * (price_shock ** 2)

            # Calculate impact from vol shock
            vega_impact = sensitivities['vega'].sum() * vol_shock

            # Total impact
            total_impact = delta_impact + gamma_impact + vega_impact
            stressed_value = baseline_value + total_impact

            results.append({
                'scenario': scenario_name,
                'baseline_value': baseline_value,
                'stressed_value': stressed_value,
                'impact': total_impact,
                'impact_percent': (total_impact / baseline_value) * 100 if baseline_value != 0 else 0
            })

        return pd.DataFrame(results)

    def plot_var_distribution(self, confidence_level: float = 0.95,
                              horizon_days: int = 1, n_simulations: int = 10000):
        """Generate and plot P&L distribution with VaR illustration"""
        if self.market_data is None:
            raise ValueError("Market data provider not set")

        # Get current portfolio value as baseline
        current_value = self.calculate_portfolio_value()

        # Simulate market factor changes
        horizon_years = horizon_days / 365.0

        # Simple portfolio simulation (same as in calculate_var)
        portfolio_volatility = 0.02

        # Generate P&L distribution
        pnl = np.random.normal(
            0, portfolio_volatility * np.sqrt(horizon_years) * current_value,
            n_simulations
        )

        # Calculate VaR
        var_index = int((1 - confidence_level) * n_simulations)
        sorted_pnl = np.sort(pnl)
        var = -sorted_pnl[var_index]

        # Plot
        plt.figure(figsize=(10, 6))

        # Plot histogram
        sns.histplot(pnl, kde=True, stat='density')

        # Mark VaR
        plt.axvline(-var, color='red', linestyle='--',
                    label=f"{confidence_level * 100:.0f}% VaR: ${var:,.2f}")

        # Shade tail
        tail_x = np.linspace(min(pnl), -var, 100)
        kde = sns.kdeplot(pnl).get_lines()[0].get_data()
        kde_x, kde_y = kde
        tail_y = np.interp(tail_x, kde_x, kde_y)
        plt.fill_between(tail_x, tail_y, alpha=0.3, color='red')

        plt.title(f"Portfolio P&L Distribution ({horizon_days}-Day Horizon)")
        plt.xlabel("Profit & Loss ($)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)

        return plt

    def plot_stress_test_results(self, stress_results: pd.DataFrame):
        """Generate waterfall chart for stress test results"""
        plt.figure(figsize=(12, 6))

        # Sort scenarios by impact
        stress_results = stress_results.sort_values('impact')

        # Create color map: red for negative impact, green for positive
        colors = ['red' if x < 0 else 'green' for x in stress_results['impact']]

        # Plot as horizontal bar chart
        plt.barh(stress_results['scenario'], stress_results['impact'], color=colors)

        plt.title('Stress Test Impact on Portfolio Value')
        plt.xlabel('Impact ($)')
        plt.ylabel('Scenario')
        plt.grid(True, axis='x')

        # Add value labels
        for i, impact in enumerate(stress_results['impact']):
            plt.text(
                impact + (np.sign(impact) * 0.01 * max(abs(stress_results['impact']))),
                i,
                f"${impact:,.2f}",
                va='center'
            )

        return plt


class EnergyForward:
    """Class for energy forward contracts"""

    def __init__(
            self,
            underlying: str,  # 'NG' or 'PW'
            delivery_location: str,
            delivery_start_date: dt.date,
            delivery_end_date: dt.date,
            contract_price: float,
            contract_size: float,
            peak_type: str = 'baseload'  # For power: 'baseload', 'peak', 'off-peak'
    ):
        self.underlying = underlying
        self.location = delivery_location
        self.delivery_start_date = delivery_start_date
        self.delivery_end_date = delivery_end_date
        self.contract_price = contract_price
        self.contract_size = contract_size

        # Power-specific attribute
        if underlying == 'PW':
            self.peak_type = peak_type

    def price(self, market_data_provider: 'MarketDataProvider',
              risk_free_rate: float, valuation_date: dt.date = None) -> float:
        """
        Calculate the mark-to-market value of the forward contract

        Args:
            market_data_provider: Provider for market data
            risk_free_rate: Risk-free rate
            valuation_date: Date for pricing (defaults to today)

        Returns:
            Mark-to-market value
        """
        if valuation_date is None:
            valuation_date = dt.date.today()

        # Get forward curve
        forward_curve = market_data_provider.get_forward_curve(
            self.underlying, self.location
        )

        # Calculate average market forward price for delivery period
        delivery_dates = pd.date_range(
            start=self.delivery_start_date,
            end=self.delivery_end_date,
            freq='D'
        )

        # Get forward prices for each delivery date
        # In a real system, this would properly interpolate the curve
        market_prices = []
        for date in delivery_dates:
            if date.date() in forward_curve.index:
                price = forward_curve.loc[date.date(), 'forward_price']
            else:
                # Find closest date
                closest_date = min(forward_curve.index,
                                   key=lambda x: abs((x - pd.Timestamp(date)).days))
                price = forward_curve.loc[closest_date, 'forward_price']

            # Apply peak adjustment for power
            if self.underlying == 'PW':
                if self.peak_type == 'peak':
                    price *= 1.3
                elif self.peak_type == 'off-peak':
                    price *= 0.7

            market_prices.append(price)

        # Average market forward price
        avg_market_price = np.mean(market_prices)

        # Calculate MTM value
        # For a forward, the value is the discounted difference between market and contract price
        delivery_period = (self.delivery_end_date - self.delivery_start_date).days + 1
        volume = self.contract_size * delivery_period

        # Time to delivery midpoint for discounting
        mid_date = self.delivery_start_date + (self.delivery_end_date - self.delivery_start_date) / 2
        time_to_delivery = (mid_date - valuation_date).days / 365.0
        discount_factor = np.exp(-risk_free_rate * time_to_delivery)

        # MTM calculation
        mtm = (avg_market_price - self.contract_price) * volume * discount_factor

        return mtm


# Example usage
if __name__ == "__main__":
    # Create a market data provider
    api_key = "your_api_key"
    market_data = MarketDataProvider(api_key)

    # Current date and some future dates
    current_date = dt.date.today()
    next_month = current_date.replace(day=1) + dt.timedelta(days=32)
    next_month = next_month.replace(day=1)
    next_quarter = next_month + dt.timedelta(days=90)

    # Create a portfolio
    portfolio = EnergyPortfolio("Energy Trading Desk")
    portfolio.set_market_data_provider(market_data)

    # Add some option positions
    # Natural gas call option
    ng_call = NaturalGasOption(
        option_type='call',
        strike=4.50,
        expiry_date=next_month,
        location='HH',
        contract_size=10000
    )
    portfolio.add_position(ng_call, 5)  # Long 5 contracts

    # Natural gas put option
    ng_put = NaturalGasOption(
        option_type='put',
        strike=4.25,
        expiry_date=next_month,
        location='HH',
        contract_size=10000
    )
    portfolio.add_position(ng_put, -3)  # Short 3 contracts

    # Power call option
    power_call = PowerOption(
        option_type='call',
        strike=55.0,
        expiry_date=next_quarter,
        location='PJM',
        peak_type='peak',
        contract_size=40
    )
    portfolio.add_position(power_call, 2)  # Long 2 contracts

    # Add a forward position
    ng_forward = EnergyForward(
        underlying='NG',
        delivery_location='HH',
        delivery_start_date=next_month,
        delivery_end_date=next_month + dt.timedelta(days=30),
        contract_price=4.35,
        contract_size=10000
    )
    portfolio.add_position(ng_forward, 1)  # Long 1 contract

    # Calculate portfolio value
    portfolio_value = portfolio.calculate_portfolio_value()
    print(f"Portfolio Value: ${portfolio_value:,.2f}")

    # Calculate VaR
    var_results = portfolio.calculate_var(confidence_level=0.95, horizon_days=1)
    print(f"1-Day 95% VaR: ${var_results['var']:,.2f}")
    print(f"1-Day 95% CVaR: ${var_results['cvar']:,.2f}")

    # Calculate sensitivities
    sensitivities = portfolio.calculate_sensitivities()
    print("\nPortfolio Sensitivities:")
    print(sensitivities)

    # Run stress tests
    stress_scenarios = [
        {'name': 'Gas Price +20%', 'price_shock': 0.20, 'vol_shock': 0},
        {'name': 'Gas Price -20%', 'price_shock': -0.20, 'vol_shock': 0},
        {'name': 'Volatility +50%', 'price_shock': 0, 'vol_shock': 0.50},
        {'name': 'Volatility -30%', 'price_shock': 0, 'vol_shock': -0.30},
        {'name': 'Combined Shock', 'price_shock': -0.15, 'vol_shock': 0.40}
    ]

    stress_results = portfolio.run_stress_test(stress_scenarios)
    print("\nStress Test Results:")
    print(stress_results)

    # Generate and save plots
    print("\nGenerating plots...")
    var_plot = portfolio.plot_var_distribution()
    var_plot.savefig("portfolio_var_distribution.png")

    stress_plot = portfolio.plot_stress_test_results(stress_results)
    stress_plot.savefig("stress_test_results.png")

    print("Analysis complete!")