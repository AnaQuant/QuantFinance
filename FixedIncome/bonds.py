import numpy as np
from scipy.optimize import newton


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
