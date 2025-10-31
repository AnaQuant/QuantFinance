"""
Black-Derman-Toy (BDT) Interest Rate Tree Model.

The BDT model is a discrete-time, no-arbitrage binomial tree model for
interest rate dynamics. It was designed to be consistent with both the
observed yield curve and the term structure of volatility.
"""

import numpy as np
from typing import List, Optional, Tuple
from .base import DiscreteTimeModel


class BlackDermanToy(DiscreteTimeModel):
    """
    Black-Derman-Toy binomial tree model for short-term interest rates.

    The BDT model assumes that the short rate follows a binomial tree where:
        r_{i+1}^{up} = r_i * exp(2σᵢ√Δt)
        r_{i+1}^{down} = r_i * exp(-2σᵢ√Δt)

    where:
        - r_i is the short rate at node i
        - σᵢ is the time-dependent volatility at step i
        - Δt is the time step size

    The tree is recombining, meaning paths that lead to the same node
    result in the same rate.

    Attributes
    ----------
    r0 : float
        Initial short rate
    volatilities : list of float
        Time-dependent volatilities for each time step
    dt : float
        Time step size
    n_steps : int
        Number of time steps
    tree : np.ndarray, optional
        Computed rate tree
    """

    def __init__(
        self,
        r0: float,
        volatilities: List[float],
        dt: float = 1.0
    ):
        """
        Initialize the Black-Derman-Toy model.

        Parameters
        ----------
        r0 : float
            Initial short rate (must be positive)
        volatilities : list of float
            Time-dependent volatilities for each time step (must be positive)
        dt : float, optional
            Time step size in years (default: 1.0)
        """
        if r0 <= 0:
            raise ValueError("r0 must be positive")
        if any(sigma <= 0 for sigma in volatilities):
            raise ValueError("All volatilities must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.r0 = r0
        self.volatilities = volatilities
        self.dt = dt
        self.n_steps = len(volatilities) + 1
        self.tree = None

    def build_tree(self, n_steps: Optional[int] = None, dt: Optional[float] = None) -> np.ndarray:
        """
        Build the binomial rate tree.

        Parameters
        ----------
        n_steps : int, optional
            Number of time steps (overrides self.n_steps if provided)
        dt : float, optional
            Time step size (overrides self.dt if provided)

        Returns
        -------
        np.ndarray
            Interest rate tree of shape (n_steps, n_steps)
            Element (j, i) represents the rate at time step i, node j
        """
        if n_steps is None:
            n_steps = self.n_steps
        if dt is None:
            dt = self.dt

        # Initialize tree
        tree = np.zeros((n_steps, n_steps))
        tree[0, 0] = self.r0

        # Build tree using log-normal dynamics
        sqrt_dt = np.sqrt(dt)

        for i in range(1, n_steps):
            sigma_i = self.volatilities[min(i - 1, len(self.volatilities) - 1)]
            for j in range(i + 1):
                # Calculate rate at node (j, i)
                # The rate depends on the number of up moves (j) and down moves (i - j)
                exponent = sigma_i * (2 * j - i) * sqrt_dt
                tree[j, i] = self.r0 * np.exp(exponent)

        self.tree = tree
        return tree

    def get_rate(self, time_step: int, node: int) -> float:
        """
        Get the interest rate at a specific node.

        Parameters
        ----------
        time_step : int
            Time step index
        node : int
            Node index at the given time step

        Returns
        -------
        float
            Interest rate at the specified node
        """
        if self.tree is None:
            self.build_tree()

        if time_step >= self.n_steps or node > time_step:
            raise ValueError("Invalid time_step or node index")

        return self.tree[node, time_step]

    def price_zero_coupon_bond(
        self,
        face_value: float = 100.0,
        maturity_steps: Optional[int] = None
    ) -> float:
        """
        Price a zero-coupon bond using the BDT tree.

        Parameters
        ----------
        face_value : float, optional
            Face value of the bond (default: 100.0)
        maturity_steps : int, optional
            Number of steps to maturity (default: n_steps)

        Returns
        -------
        float
            Present value of the zero-coupon bond
        """
        if self.tree is None:
            self.build_tree()

        if maturity_steps is None:
            maturity_steps = self.n_steps

        # Initialize bond prices at maturity
        bond_prices = np.zeros((maturity_steps, maturity_steps))
        bond_prices[:, -1] = face_value

        # Backward induction
        for i in range(maturity_steps - 2, -1, -1):
            for j in range(i + 1):
                r = self.tree[j, i]
                discount_factor = np.exp(-r * self.dt)
                # Risk-neutral valuation: equal probability of up/down
                bond_prices[j, i] = discount_factor * 0.5 * \
                    (bond_prices[j, i + 1] + bond_prices[j + 1, i + 1])

        return bond_prices[0, 0]

    def price_coupon_bond(
        self,
        face_value: float,
        coupon_rate: float,
        maturity_steps: Optional[int] = None
    ) -> float:
        """
        Price a coupon-bearing bond using the BDT tree.

        Parameters
        ----------
        face_value : float
            Face value of the bond
        coupon_rate : float
            Annual coupon rate (as a decimal, e.g., 0.05 for 5%)
        maturity_steps : int, optional
            Number of steps to maturity (default: n_steps)

        Returns
        -------
        float
            Present value of the coupon bond
        """
        if self.tree is None:
            self.build_tree()

        if maturity_steps is None:
            maturity_steps = self.n_steps

        # Calculate coupon payment per period
        coupon_payment = coupon_rate * face_value * self.dt

        # Initialize bond prices at maturity (including final coupon)
        bond_prices = np.zeros((maturity_steps, maturity_steps))
        bond_prices[:, -1] = face_value + coupon_payment

        # Backward induction
        for i in range(maturity_steps - 2, -1, -1):
            for j in range(i + 1):
                r = self.tree[j, i]
                discount_factor = np.exp(-r * self.dt)
                # Risk-neutral valuation with coupon payment
                bond_prices[j, i] = discount_factor * 0.5 * \
                    (bond_prices[j, i + 1] + bond_prices[j + 1, i + 1]) + \
                    coupon_payment

        return bond_prices[0, 0]

    def get_forward_rates(self, time_step: int) -> np.ndarray:
        """
        Get all forward rates at a specific time step.

        Parameters
        ----------
        time_step : int
            Time step index

        Returns
        -------
        np.ndarray
            Array of forward rates at the given time step
        """
        if self.tree is None:
            self.build_tree()

        if time_step >= self.n_steps:
            raise ValueError("time_step exceeds tree depth")

        return self.tree[:time_step + 1, time_step]

    def calibrate_volatilities(
        self,
        market_bond_prices: List[float],
        face_value: float = 100.0
    ) -> List[float]:
        """
        Calibrate volatilities to match market bond prices.

        This is a simplified calibration routine. A full calibration would
        require numerical optimization to fit the term structure.

        Parameters
        ----------
        market_bond_prices : list of float
            Observed market prices for zero-coupon bonds of different maturities
        face_value : float, optional
            Face value of bonds (default: 100.0)

        Returns
        -------
        list of float
            Calibrated volatilities
        """
        # Placeholder for calibration logic
        # Full implementation would use numerical methods to find volatilities
        # that reproduce market prices
        raise NotImplementedError(
            "Volatility calibration requires numerical optimization. "
            "Provide volatilities directly or implement custom calibration."
        )


# Alias for convenience
BDT = BlackDermanToy
