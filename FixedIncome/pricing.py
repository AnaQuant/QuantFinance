"""
Advanced Bond Pricing with Interest Rate Models.

This module provides bond pricing functionality using stochastic interest rate models
from the StochasticProcesses package.
"""

import numpy as np
from typing import Optional
import sys
import os

# Add parent directory to path to import StochasticProcesses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .bonds import Bond, CallableBond


class BondPricerWithIRModel:
    """
    Bond pricer using stochastic interest rate models.

    This class integrates bond pricing with interest rate models like
    CIR and Black-Derman-Toy.
    """

    def __init__(self, ir_model):
        """
        Initialize the bond pricer with an interest rate model.

        Parameters
        ----------
        ir_model : object
            Interest rate model (CIRProcess or BlackDermanToy from StochasticProcesses)
        """
        self.ir_model = ir_model

    def price_with_cir(
        self,
        bond: Bond,
        n_paths: int = 10000,
        n_steps: int = 100
    ) -> dict:
        """
        Price a bond using Monte Carlo simulation with the CIR model.

        Parameters
        ----------
        bond : Bond
            Bond object to price
        n_paths : int, optional
            Number of Monte Carlo paths (default: 10000)
        n_steps : int, optional
            Number of time steps (default: 100)

        Returns
        -------
        dict
            Dictionary with 'price', 'std_error', and 'confidence_interval'
        """
        from StochasticProcesses import CIRProcess

        if not isinstance(self.ir_model, CIRProcess):
            raise TypeError("ir_model must be a CIRProcess instance")

        # Simulate interest rate paths
        t, rate_paths = self.ir_model.simulate(
            T=bond.maturity,
            n_steps=n_steps,
            n_paths=n_paths
        )

        # Get bond cash flows
        cash_flows = bond.get_cash_flows()

        # Calculate present value for each path
        present_values = np.zeros(n_paths)

        for path_idx in range(n_paths):
            pv = 0.0
            for cf in cash_flows:
                # Find the time index closest to the cash flow time
                time_idx = np.argmin(np.abs(t - cf.time))
                # Average rate from 0 to cf.time for this path
                avg_rate = np.mean(rate_paths[path_idx, :time_idx + 1])
                # Discount cash flow
                discount_factor = np.exp(-avg_rate * cf.time)
                pv += cf.amount * discount_factor

            present_values[path_idx] = pv

        # Calculate statistics
        price = np.mean(present_values)
        std_error = np.std(present_values) / np.sqrt(n_paths)
        ci_95 = 1.96 * std_error

        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (price - ci_95, price + ci_95)
        }

    def price_with_bdt(
        self,
        bond: Bond
    ) -> float:
        """
        Price a bond using the Black-Derman-Toy binomial tree model.

        Parameters
        ----------
        bond : Bond
            Bond object to price

        Returns
        -------
        float
            Bond price
        """
        from StochasticProcesses import BlackDermanToy

        if not isinstance(self.ir_model, BlackDermanToy):
            raise TypeError("ir_model must be a BlackDermanToy instance")

        # Use the BDT model's built-in bond pricing
        if bond.coupon_rate == 0:
            # Zero-coupon bond
            return self.ir_model.price_zero_coupon_bond(
                face_value=bond.face_value,
                maturity_steps=bond.periods
            )
        else:
            # Coupon bond
            return self.ir_model.price_coupon_bond(
                face_value=bond.face_value,
                coupon_rate=bond.coupon_rate,
                maturity_steps=bond.periods
            )

    def price(
        self,
        bond: Bond,
        **kwargs
    ) -> dict:
        """
        Price a bond using the configured interest rate model.

        Automatically selects the appropriate pricing method based on the model type.

        Parameters
        ----------
        bond : Bond
            Bond object to price
        **kwargs
            Additional arguments passed to the specific pricing method

        Returns
        -------
        dict or float
            Pricing result (format depends on the model)
        """
        model_name = self.ir_model.__class__.__name__

        if 'CIR' in model_name:
            return self.price_with_cir(bond, **kwargs)
        elif 'BlackDermanToy' in model_name or 'BDT' in model_name:
            result = self.price_with_bdt(bond)
            return {'price': result}
        else:
            raise ValueError(f"Unsupported interest rate model: {model_name}")


def compare_pricing_methods(
    bond: Bond,
    market_rate: float,
    ir_model: Optional[object] = None,
    **kwargs
) -> dict:
    """
    Compare bond pricing using different methods.

    Parameters
    ----------
    bond : Bond
        Bond object to price
    market_rate : float
        Flat market rate for closed-form pricing
    ir_model : object, optional
        Interest rate model for stochastic pricing
    **kwargs
        Additional arguments for stochastic pricing

    Returns
    -------
    dict
        Dictionary with prices from different methods
    """
    results = {}

    # Closed-form price
    results['closed_form'] = bond.price(market_rate)

    # Stochastic model price (if provided)
    if ir_model is not None:
        pricer = BondPricerWithIRModel(ir_model)
        stochastic_result = pricer.price(bond, **kwargs)

        if isinstance(stochastic_result, dict):
            results['stochastic'] = stochastic_result['price']
            if 'std_error' in stochastic_result:
                results['stochastic_stderr'] = stochastic_result['std_error']
        else:
            results['stochastic'] = stochastic_result

        # Calculate difference
        results['difference'] = results['stochastic'] - results['closed_form']
        results['relative_difference'] = (results['difference'] /
                                         results['closed_form'] * 100)

    return results
