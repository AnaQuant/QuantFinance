"""
Shared pytest fixtures for QuantFinance test suite.

Provides a reusable Bond instance representing a standard 10-year gilt proxy,
plus module-level constants so individual test modules can reference the same
benchmark parameters without hard-coding them.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from FixedIncome.bonds import Bond

# ---------------------------------------------------------------------------
# Benchmark parameters — 10-year gilt proxy
# ---------------------------------------------------------------------------
FACE = 100.0
COUPON = 0.04   # 4 % annual coupon rate
YTM = 0.045     # 4.5 % yield to maturity
MATURITY = 10   # 10 years (integer — required by standalone bond_price)


@pytest.fixture
def bond() -> Bond:
    """
    Standard 10-year gilt proxy: face=100, coupon=4 %, frequency=annual.

    Used as a shared baseline across Bond-class tests so every test
    starts from an identical, financially meaningful instrument.
    """
    return Bond(face_value=FACE, coupon_rate=COUPON, maturity=MATURITY)
