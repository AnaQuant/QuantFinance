"""
Bond Pricing and Analysis Module.

This module provides both functional and object-oriented interfaces for
bond pricing, yield calculations, and risk metrics.
"""

import numpy as np
from scipy.optimize import newton
from typing import Optional, List
from dataclasses import dataclass


# ============================================================================
# Object-Oriented Interface (Recommended)
# ============================================================================

@dataclass
class CashFlow:
    """Represents a single cash flow."""
    time: float  # Time in years
    amount: float  # Cash flow amount


class Bond:
    """
    Base class for bond instruments.

    Attributes
    ----------
    face_value : float
        Face value (par value) of the bond
    coupon_rate : float
        Annual coupon rate (as decimal, e.g., 0.05 for 5%)
    maturity : float
        Time to maturity in years
    frequency : int
        Number of coupon payments per year (1=annual, 2=semi-annual, 4=quarterly)
    """

    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        maturity: float,
        frequency: int = 1
    ):
        """
        Initialize a Bond.

        Parameters
        ----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (decimal)
        maturity : float
            Time to maturity in years
        frequency : int, optional
            Coupon payment frequency per year (default: 1)
        """
        if face_value <= 0:
            raise ValueError("face_value must be positive")
        if coupon_rate < 0:
            raise ValueError("coupon_rate must be non-negative")
        if maturity <= 0:
            raise ValueError("maturity must be positive")
        if frequency not in [1, 2, 4, 12]:
            raise ValueError("frequency must be 1, 2, 4, or 12")

        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.frequency = frequency

    @property
    def periods(self) -> int:
        """Total number of coupon payment periods."""
        return int(self.maturity * self.frequency)

    @property
    def coupon_payment(self) -> float:
        """Cash flow amount per coupon payment."""
        return (self.coupon_rate * self.face_value) / self.frequency

    def get_cash_flows(self) -> List[CashFlow]:
        """
        Get all cash flows from the bond.

        Returns
        -------
        list of CashFlow
            List of cash flow objects
        """
        cash_flows = []
        dt = 1.0 / self.frequency

        # Coupon payments
        for i in range(1, self.periods + 1):
            cash_flows.append(CashFlow(
                time=i * dt,
                amount=self.coupon_payment
            ))

        # Add face value to final payment
        cash_flows[-1].amount += self.face_value

        return cash_flows

    def price(self, market_rate: float) -> float:
        """
        Calculate the present value of the bond.

        Parameters
        ----------
        market_rate : float
            Market interest rate (annual, as decimal)

        Returns
        -------
        float
            Bond price
        """
        cash_flows = self.get_cash_flows()
        present_value = 0.0

        for cf in cash_flows:
            discount_factor = (1 + market_rate / self.frequency) ** (-cf.time * self.frequency)
            present_value += cf.amount * discount_factor

        return present_value

    def yield_to_maturity(self, price: float, initial_guess: float = 0.05) -> Optional[float]:
        """
        Calculate the yield to maturity given a bond price.

        Parameters
        ----------
        price : float
            Current bond price
        initial_guess : float, optional
            Initial guess for YTM (default: 0.05)

        Returns
        -------
        float or None
            Yield to maturity, or None if calculation fails
        """
        def ytm_function(y):
            return self.price(y) - price

        try:
            return newton(ytm_function, initial_guess)
        except RuntimeError:
            return None

    def duration(self, market_rate: float) -> float:
        """
        Calculate Macaulay duration.

        Parameters
        ----------
        market_rate : float
            Market interest rate (annual, as decimal)

        Returns
        -------
        float
            Macaulay duration in years
        """
        cash_flows = self.get_cash_flows()
        bond_price = self.price(market_rate)

        weighted_time = 0.0
        for cf in cash_flows:
            discount_factor = (1 + market_rate / self.frequency) ** (-cf.time * self.frequency)
            weighted_time += cf.time * cf.amount * discount_factor

        return weighted_time / bond_price

    def modified_duration(self, market_rate: float) -> float:
        """
        Calculate modified duration.

        Modified duration measures the price sensitivity to yield changes.

        Parameters
        ----------
        market_rate : float
            Market interest rate (annual, as decimal)

        Returns
        -------
        float
            Modified duration
        """
        mac_duration = self.duration(market_rate)
        return mac_duration / (1 + market_rate / self.frequency)

    def convexity(self, market_rate: float) -> float:
        """
        Calculate convexity.

        Convexity measures the curvature of the price-yield relationship.

        Parameters
        ----------
        market_rate : float
            Market interest rate (annual, as decimal)

        Returns
        -------
        float
            Convexity
        """
        cash_flows = self.get_cash_flows()
        bond_price = self.price(market_rate)

        convexity_sum = 0.0
        for cf in cash_flows:
            t = cf.time * self.frequency
            discount_factor = (1 + market_rate / self.frequency) ** (-t)
            convexity_sum += cf.amount * t * (t + 1) * discount_factor

        return convexity_sum / (bond_price * (1 + market_rate / self.frequency) ** 2)

    def __repr__(self) -> str:
        return (f"Bond(face_value={self.face_value}, coupon_rate={self.coupon_rate:.4f}, "
                f"maturity={self.maturity}, frequency={self.frequency})")


class CallableBond(Bond):
    """
    Callable bond that can be redeemed early by the issuer.

    Attributes
    ----------
    call_price : float
        Price at which the bond can be called
    call_date : float
        Earliest time (in years) when the bond can be called
    """

    def __init__(
        self,
        face_value: float,
        coupon_rate: float,
        maturity: float,
        call_price: float,
        call_date: float,
        frequency: int = 1
    ):
        """
        Initialize a callable bond.

        Parameters
        ----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (decimal)
        maturity : float
            Time to maturity in years
        call_price : float
            Price at which the bond can be called
        call_date : float
            Earliest call date in years
        frequency : int, optional
            Coupon payment frequency per year (default: 1)
        """
        super().__init__(face_value, coupon_rate, maturity, frequency)

        if call_date >= maturity:
            raise ValueError("call_date must be before maturity")
        if call_price <= 0:
            raise ValueError("call_price must be positive")

        self.call_price = call_price
        self.call_date = call_date

    def price(self, market_rate: float) -> float:
        """
        Calculate the price of the callable bond.

        Uses a simplified approach: takes the minimum of the regular bond price
        and the present value of the call price.

        Parameters
        ----------
        market_rate : float
            Market interest rate (annual, as decimal)

        Returns
        -------
        float
            Callable bond price
        """
        regular_price = super().price(market_rate)

        # Present value of call price
        call_periods = self.call_date * self.frequency
        call_price_pv = self.call_price / (1 + market_rate / self.frequency) ** call_periods

        # Bond is worth the minimum (issuer will call if beneficial)
        return min(regular_price, call_price_pv)

    def yield_to_call(self, price: float, initial_guess: float = 0.05) -> Optional[float]:
        """
        Calculate the yield to call.

        Parameters
        ----------
        price : float
            Current bond price
        initial_guess : float, optional
            Initial guess for YTC (default: 0.05)

        Returns
        -------
        float or None
            Yield to call, or None if calculation fails
        """
        # Create a temporary bond with maturity at call date and face value = call price
        temp_bond = Bond(
            face_value=self.call_price,
            coupon_rate=self.coupon_rate,
            maturity=self.call_date,
            frequency=self.frequency
        )

        return temp_bond.yield_to_maturity(price, initial_guess)

    def __repr__(self) -> str:
        return (f"CallableBond(face_value={self.face_value}, coupon_rate={self.coupon_rate:.4f}, "
                f"maturity={self.maturity}, call_price={self.call_price}, "
                f"call_date={self.call_date}, frequency={self.frequency})")


# ============================================================================
# Functional Interface (Legacy, kept for backwards compatibility)
# ============================================================================

def bond_price(face_value, coupon_rate, periods, market_rate, frequency=1):
    """
    Calculate the price of a bond using the present value of cash flows.
    :param face_value: Face value of the bond (e.g., 1000)
    :param coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
    :param periods: Total number of periods until maturity
    :param market_rate: Market interest rate per period (decimal form)
    :param frequency: Number of coupon payments per year
    :return: Bond price
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    discount_factors = [(1 + market_rate / frequency) ** -(i + 1) for i in range(periods)]
    present_value_coupons = sum(coupon_payment * df for df in discount_factors)
    present_value_face = face_value * discount_factors[-1]
    return present_value_coupons + present_value_face


def yield_to_maturity(price, face_value, coupon_rate, periods, frequency=1):
    """
    Estimate the yield to maturity (YTM) using numerical root-finding.
    :param price: Current bond price
    :param face_value: Face value of the bond
    :param coupon_rate: Annual coupon rate
    :param periods: Total number of periods until maturity
    :param frequency: Number of coupon payments per year
    :return: Yield to maturity (YTM) per period
    """

    def ytm_function(y):
        return sum([(coupon_rate * face_value / frequency) / (1 + y / frequency) ** (i + 1) for i in range(periods)]) \
            + face_value / (1 + y / frequency) ** periods - price

    try:
        return newton(ytm_function, 0.05)  # Initial guess of 5%
    except RuntimeError:
        return None  # Return None if YTM calculation fails


def callable_bond_price(face_value, coupon_rate, periods, market_rate, call_price, call_period, frequency=1):
    """
    Calculate the price of a callable bond, assuming it may be called at a given call price.
    :param face_value: Face value of the bond
    :param coupon_rate: Annual coupon rate
    :param periods: Total number of periods until maturity
    :param market_rate: Market interest rate per period
    :param call_price: Price at which the bond can be called
    :param call_period: Period when the bond can be called
    :param frequency: Number of coupon payments per year
    :return: Callable bond price
    """
    regular_price = bond_price(face_value, coupon_rate, periods, market_rate, frequency)
    call_price_discounted = call_price / (1 + market_rate / frequency) ** call_period
    return min(regular_price, call_price_discounted)


def bond_duration(price, face_value, coupon_rate, periods, market_rate, frequency=1):
    """
    Calculate the Macaulay duration of a bond.
    :param price: Current bond price
    :param face_value: Face value of the bond
    :param coupon_rate: Annual coupon rate
    :param periods: Total number of periods until maturity
    :param market_rate: Market interest rate per period
    :param frequency: Number of coupon payments per year
    :return: Macaulay duration
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    discount_factors = [(1 + market_rate / frequency) ** -(i + 1) for i in range(periods)]
    weighted_cash_flows = [(i + 1) * coupon_payment * df for i, df in enumerate(discount_factors)]
    weighted_cash_flows.append(periods * face_value * discount_factors[-1])
    return sum(weighted_cash_flows) / price


# Example usage
if __name__ == "__main__":
    face_value = 1000  # Bond's face value
    coupon_rate = 0.05  # 5% annual coupon
    periods = 10  # Number of periods (e.g., 10 years)
    market_rate = 0.04  # 4% market rate
    call_price = 1050  # Callable at $1050
    call_period = 5  # Callable in 5 years
    frequency = 1  # Annual payments

    price = bond_price(face_value, coupon_rate, periods, market_rate, frequency)
    print(f"Bond Price: {price:.2f}")

    ytm = yield_to_maturity(price, face_value, coupon_rate, periods, frequency)
    print(f"Yield to Maturity: {ytm:.6f}")

    callable_price = callable_bond_price(face_value, coupon_rate, periods, market_rate, call_price, call_period,
                                         frequency)
    print(f"Callable Bond Price: {callable_price:.2f}")

    duration = bond_duration(price, face_value, coupon_rate, periods, market_rate, frequency)
    print(f"Macaulay Duration: {duration:.4f}")
