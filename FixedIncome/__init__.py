"""
Fixed Income Package

Comprehensive toolkit for bond pricing, yield curve analysis, and interest rate modeling.
"""

from .bonds import Bond, CallableBond, CashFlow
from .yield_curve import YieldCurve

# Import legacy functions for backwards compatibility
from .bonds import (
    bond_price,
    yield_to_maturity,
    callable_bond_price,
    bond_duration
)

__all__ = [
    # Classes
    'Bond',
    'CallableBond',
    'CashFlow',
    'YieldCurve',
    # Legacy functions
    'bond_price',
    'yield_to_maturity',
    'callable_bond_price',
    'bond_duration',
]

__version__ = '0.1.0'
