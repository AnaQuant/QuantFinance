"""
Yield Curve Management and Analysis.

This module provides classes for managing, storing, and analyzing yield curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple, Dict
from datetime import datetime


class YieldCurve:
    """
    Yield curve class for managing and analyzing interest rate term structures.

    Attributes
    ----------
    tenors : np.ndarray
        Array of maturities (in years)
    yields : np.ndarray
        Array of yields (as decimals, e.g., 0.05 for 5%)
    date : datetime, optional
        Date of the yield curve observation
    currency : str, optional
        Currency code (e.g., 'USD', 'EUR')
    interpolator : object, optional
        Interpolation object (from interpolation.py)
    """

    def __init__(
        self,
        tenors: Union[List[float], np.ndarray],
        yields: Union[List[float], np.ndarray],
        date: Optional[datetime] = None,
        currency: Optional[str] = None
    ):
        """
        Initialize a yield curve.

        Parameters
        ----------
        tenors : array-like
            Maturities in years
        yields : array-like
            Yields as decimals (e.g., 0.05 for 5%)
        date : datetime, optional
            Date of observation
        currency : str, optional
            Currency code
        """
        self.tenors = np.array(tenors, dtype=float)
        self.yields = np.array(yields, dtype=float)

        if len(self.tenors) != len(self.yields):
            raise ValueError("tenors and yields must have the same length")

        # Sort by tenors
        sort_idx = np.argsort(self.tenors)
        self.tenors = self.tenors[sort_idx]
        self.yields = self.yields[sort_idx]

        self.date = date
        self.currency = currency
        self.interpolator = None

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        tenor_col: str = 'tenor',
        yield_col: str = 'yield',
        date: Optional[datetime] = None,
        currency: Optional[str] = None
    ) -> 'YieldCurve':
        """
        Create a YieldCurve from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing tenor and yield columns
        tenor_col : str
            Name of the tenor column
        yield_col : str
            Name of the yield column
        date : datetime, optional
            Date of observation
        currency : str, optional
            Currency code

        Returns
        -------
        YieldCurve
            New YieldCurve instance
        """
        return cls(
            tenors=df[tenor_col].values,
            yields=df[yield_col].values,
            date=date,
            currency=currency
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[float, float],
        date: Optional[datetime] = None,
        currency: Optional[str] = None
    ) -> 'YieldCurve':
        """
        Create a YieldCurve from a dictionary mapping tenors to yields.

        Parameters
        ----------
        data : dict
            Dictionary with tenors as keys and yields as values
        date : datetime, optional
            Date of observation
        currency : str, optional
            Currency code

        Returns
        -------
        YieldCurve
            New YieldCurve instance
        """
        tenors = list(data.keys())
        yields = list(data.values())
        return cls(tenors, yields, date, currency)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert yield curve to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with tenor and yield columns
        """
        df = pd.DataFrame({
            'tenor': self.tenors,
            'yield': self.yields
        })

        if self.date is not None:
            df['date'] = self.date
        if self.currency is not None:
            df['currency'] = self.currency

        return df

    def get_yield(self, tenor: float, interpolate: bool = True) -> float:
        """
        Get the yield for a specific tenor.

        Parameters
        ----------
        tenor : float
            Maturity in years
        interpolate : bool, optional
            Whether to interpolate if tenor is not in the curve (default: True)

        Returns
        -------
        float
            Yield at the specified tenor
        """
        # Check if tenor exists exactly
        idx = np.where(np.isclose(self.tenors, tenor))[0]
        if len(idx) > 0:
            return self.yields[idx[0]]

        if not interpolate:
            raise ValueError(f"Tenor {tenor} not found in curve and interpolate=False")

        if self.interpolator is None:
            # Use simple linear interpolation as fallback
            return np.interp(tenor, self.tenors, self.yields)

        return self.interpolator.interpolate(tenor)

    def get_discount_factor(self, tenor: float) -> float:
        """
        Calculate the discount factor for a given tenor.

        Discount factor: DF(t) = exp(-y(t) * t)

        Parameters
        ----------
        tenor : float
            Maturity in years

        Returns
        -------
        float
            Discount factor
        """
        y = self.get_yield(tenor)
        return np.exp(-y * tenor)

    def get_forward_rate(self, t1: float, t2: float) -> float:
        """
        Calculate the forward rate between two points in time.

        Forward rate: f(t1, t2) = (y(t2)*t2 - y(t1)*t1) / (t2 - t1)

        Parameters
        ----------
        t1 : float
            Start time in years
        t2 : float
            End time in years

        Returns
        -------
        float
            Forward rate
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")

        y1 = self.get_yield(t1)
        y2 = self.get_yield(t2)

        return (y2 * t2 - y1 * t1) / (t2 - t1)

    def get_spot_rate_curve(self, tenors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get spot rates for specified tenors.

        Parameters
        ----------
        tenors : np.ndarray, optional
            Tenors to compute spot rates for (default: use existing tenors)

        Returns
        -------
        tenors : np.ndarray
            Tenor points
        spot_rates : np.ndarray
            Spot rates
        """
        if tenors is None:
            return self.tenors.copy(), self.yields.copy()

        spot_rates = np.array([self.get_yield(t) for t in tenors])
        return tenors, spot_rates

    def plot(
        self,
        title: Optional[str] = None,
        xlabel: str = "Maturity (Years)",
        ylabel: str = "Yield (%)",
        figsize: Tuple[int, int] = (10, 6),
        show_points: bool = True,
        show_grid: bool = True
    ) -> None:
        """
        Plot the yield curve.

        Parameters
        ----------
        title : str, optional
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        figsize : tuple
            Figure size
        show_points : bool
            Whether to show data points
        show_grid : bool
            Whether to show grid
        """
        plt.figure(figsize=figsize)

        # Plot interpolated curve if interpolator exists
        if self.interpolator is not None:
            fine_tenors = np.linspace(self.tenors.min(), self.tenors.max(), 200)
            fine_yields = np.array([self.interpolator.interpolate(t) for t in fine_tenors])
            plt.plot(fine_tenors, fine_yields * 100, '-', label='Interpolated Curve')

        # Plot actual data points
        if show_points:
            marker = 'o' if self.interpolator is not None else 'o-'
            plt.plot(self.tenors, self.yields * 100, marker, label='Market Data')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is None:
            if self.date is not None and self.currency is not None:
                title = f"{self.currency} Yield Curve - {self.date.strftime('%Y-%m-%d')}"
            elif self.currency is not None:
                title = f"{self.currency} Yield Curve"
            elif self.date is not None:
                title = f"Yield Curve - {self.date.strftime('%Y-%m-%d')}"
            else:
                title = "Yield Curve"

        plt.title(title)

        if self.interpolator is not None or show_points:
            plt.legend()

        if show_grid:
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def set_interpolator(self, interpolator) -> None:
        """
        Set the interpolation method for the yield curve.

        Parameters
        ----------
        interpolator : object
            Interpolator object (from interpolation.py)
        """
        self.interpolator = interpolator
        # Fit the interpolator to the current data
        if hasattr(interpolator, 'fit'):
            interpolator.fit(self.tenors, self.yields)

    def get_shape_metrics(self) -> Dict[str, float]:
        """
        Calculate various metrics describing the shape of the yield curve.

        Returns
        -------
        dict
            Dictionary containing shape metrics:
            - slope_2_10: 10-year yield minus 2-year yield
            - slope_3m_10: 10-year yield minus 3-month yield
            - level: Average yield across all tenors
            - curvature: 2*(5-year yield) - (2-year yield) - (10-year yield)
        """
        metrics = {}

        # Level (average yield)
        metrics['level'] = np.mean(self.yields)

        # Slopes (if tenors exist)
        try:
            y_2y = self.get_yield(2.0)
            y_10y = self.get_yield(10.0)
            metrics['slope_2_10'] = y_10y - y_2y
        except:
            metrics['slope_2_10'] = np.nan

        try:
            y_3m = self.get_yield(0.25)
            y_10y = self.get_yield(10.0)
            metrics['slope_3m_10'] = y_10y - y_3m
        except:
            metrics['slope_3m_10'] = np.nan

        # Curvature (butterfly)
        try:
            y_2y = self.get_yield(2.0)
            y_5y = self.get_yield(5.0)
            y_10y = self.get_yield(10.0)
            metrics['curvature'] = 2 * y_5y - y_2y - y_10y
        except:
            metrics['curvature'] = np.nan

        return metrics

    def __repr__(self) -> str:
        """String representation of the yield curve."""
        date_str = f", date={self.date.strftime('%Y-%m-%d')}" if self.date else ""
        currency_str = f", currency={self.currency}" if self.currency else ""
        return f"YieldCurve(n_points={len(self.tenors)}{date_str}{currency_str})"
