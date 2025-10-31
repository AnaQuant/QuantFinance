"""
Stochastic Processes for Quantitative Finance

This package provides implementations of various stochastic processes used in
quantitative finance, including:
- Geometric Brownian Motion (GBM)
- Ornstein-Uhlenbeck (OU) process
- Cox-Ingersoll-Ross (CIR) process
- Heston stochastic volatility model
- Black-Derman-Toy (BDT) interest rate tree model
"""

from .geometric_brownian_motion import GeometricBrownianMotion, GBM
from .ornstein_uhlenbeck import OrnsteinUhlenbeck, OU
from .cir_process import CIRProcess
from .heston_model import HestonModel
from .black_derman_toy import BlackDermanToy, BDT

__all__ = [
    'GeometricBrownianMotion',
    'GBM',
    'OrnsteinUhlenbeck',
    'OU',
    'CIRProcess',
    'HestonModel',
    'BlackDermanToy',
    'BDT',
]

__version__ = '0.1.0'
