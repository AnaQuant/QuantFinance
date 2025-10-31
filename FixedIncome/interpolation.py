"""
Yield Curve Interpolation Methods.

This module provides various interpolation methods for yield curves, including:
- Cubic spline interpolation
- Nelson-Siegel model
- Nelson-Siegel-Svensson model
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class Interpolator(ABC):
    """
    Abstract base class for yield curve interpolators.
    """

    @abstractmethod
    def fit(self, tenors: np.ndarray, yields: np.ndarray) -> None:
        """Fit the interpolator to data."""
        pass

    @abstractmethod
    def interpolate(self, tenor: float) -> float:
        """Interpolate yield at a given tenor."""
        pass

    def interpolate_multiple(self, tenors: np.ndarray) -> np.ndarray:
        """Interpolate yields at multiple tenors."""
        return np.array([self.interpolate(t) for t in tenors])


class CubicSplineInterpolator(Interpolator):
    """
    Cubic spline interpolation for yield curves.

    This interpolator uses scipy's CubicSpline to provide smooth
    interpolation between observed points.

    Attributes
    ----------
    spline : CubicSpline
        Fitted cubic spline object
    """

    def __init__(self, bc_type: str = 'not-a-knot'):
        """
        Initialize the cubic spline interpolator.

        Parameters
        ----------
        bc_type : str, optional
            Boundary condition type for the spline. Options:
            - 'not-a-knot' (default)
            - 'natural'
            - 'clamped'
        """
        self.bc_type = bc_type
        self.spline = None

    def fit(self, tenors: np.ndarray, yields: np.ndarray) -> None:
        """
        Fit the cubic spline to the yield curve data.

        Parameters
        ----------
        tenors : np.ndarray
            Observed tenors
        yields : np.ndarray
            Observed yields
        """
        self.spline = CubicSpline(tenors, yields, bc_type=self.bc_type)

    def interpolate(self, tenor: float) -> float:
        """
        Interpolate the yield at a given tenor.

        Parameters
        ----------
        tenor : float
            Tenor to interpolate at

        Returns
        -------
        float
            Interpolated yield
        """
        if self.spline is None:
            raise ValueError("Interpolator has not been fitted. Call fit() first.")

        return float(self.spline(tenor))


class NelsonSiegelInterpolator(Interpolator):
    """
    Nelson-Siegel parametric model for yield curve interpolation.

    The Nelson-Siegel model represents the yield curve as:
        y(t) = β₀ + β₁ * [(1 - exp(-t/τ))/(t/τ)] +
               β₂ * [(1 - exp(-t/τ))/(t/τ) - exp(-t/τ)]

    where:
        - β₀: long-term level
        - β₁: short-term component
        - β₂: medium-term component (curvature)
        - τ: decay factor

    Attributes
    ----------
    beta0 : float
        Long-term level parameter
    beta1 : float
        Short-term component parameter
    beta2 : float
        Medium-term component parameter
    tau : float
        Decay factor parameter
    """

    def __init__(self):
        """Initialize the Nelson-Siegel interpolator."""
        self.beta0 = None
        self.beta1 = None
        self.beta2 = None
        self.tau = None

    @staticmethod
    def _nelson_siegel(t: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
        """
        Nelson-Siegel functional form.

        Parameters
        ----------
        t : float
            Maturity
        beta0, beta1, beta2, tau : float
            Model parameters

        Returns
        -------
        float
            Yield at maturity t
        """
        if t == 0:
            return beta0 + beta1

        factor = (1 - np.exp(-t / tau)) / (t / tau)
        return beta0 + beta1 * factor + beta2 * (factor - np.exp(-t / tau))

    def fit(
        self,
        tenors: np.ndarray,
        yields: np.ndarray,
        initial_guess: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """
        Fit the Nelson-Siegel model to yield curve data.

        Parameters
        ----------
        tenors : np.ndarray
            Observed tenors
        yields : np.ndarray
            Observed yields
        initial_guess : tuple, optional
            Initial parameter guess (beta0, beta1, beta2, tau)
            Default: (mean(yields), -0.01, 0.01, 2.0)
        """
        if initial_guess is None:
            initial_guess = (np.mean(yields), -0.01, 0.01, 2.0)

        # Vectorized Nelson-Siegel function
        def ns_vec(t, beta0, beta1, beta2, tau):
            return np.array([self._nelson_siegel(ti, beta0, beta1, beta2, tau) for ti in t])

        # Fit using curve_fit
        try:
            params, _ = curve_fit(
                ns_vec,
                tenors,
                yields,
                p0=initial_guess,
                maxfev=10000
            )
            self.beta0, self.beta1, self.beta2, self.tau = params
        except RuntimeError:
            raise RuntimeError("Nelson-Siegel fitting failed to converge")

    def interpolate(self, tenor: float) -> float:
        """
        Interpolate the yield at a given tenor using the fitted Nelson-Siegel model.

        Parameters
        ----------
        tenor : float
            Tenor to interpolate at

        Returns
        -------
        float
            Interpolated yield
        """
        if self.beta0 is None:
            raise ValueError("Interpolator has not been fitted. Call fit() first.")

        return self._nelson_siegel(tenor, self.beta0, self.beta1, self.beta2, self.tau)

    def get_parameters(self) -> dict:
        """
        Get the fitted model parameters.

        Returns
        -------
        dict
            Dictionary containing beta0, beta1, beta2, and tau
        """
        return {
            'beta0': self.beta0,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'tau': self.tau
        }


class NelsonSiegelSvenssonInterpolator(Interpolator):
    """
    Nelson-Siegel-Svensson extended parametric model.

    The NSS model extends Nelson-Siegel with an additional term:
        y(t) = β₀ + β₁ * [(1 - exp(-t/τ₁))/(t/τ₁)] +
               β₂ * [(1 - exp(-t/τ₁))/(t/τ₁) - exp(-t/τ₁)] +
               β₃ * [(1 - exp(-t/τ₂))/(t/τ₂) - exp(-t/τ₂)]

    This additional term allows for more flexible curve shapes.

    Attributes
    ----------
    beta0, beta1, beta2, beta3 : float
        Shape parameters
    tau1, tau2 : float
        Decay factors
    """

    def __init__(self):
        """Initialize the Nelson-Siegel-Svensson interpolator."""
        self.beta0 = None
        self.beta1 = None
        self.beta2 = None
        self.beta3 = None
        self.tau1 = None
        self.tau2 = None

    @staticmethod
    def _nss(t: float, beta0: float, beta1: float, beta2: float,
             beta3: float, tau1: float, tau2: float) -> float:
        """
        Nelson-Siegel-Svensson functional form.

        Parameters
        ----------
        t : float
            Maturity
        beta0, beta1, beta2, beta3, tau1, tau2 : float
            Model parameters

        Returns
        -------
        float
            Yield at maturity t
        """
        if t == 0:
            return beta0 + beta1

        factor1 = (1 - np.exp(-t / tau1)) / (t / tau1)
        factor2 = (1 - np.exp(-t / tau2)) / (t / tau2)

        return (beta0 +
                beta1 * factor1 +
                beta2 * (factor1 - np.exp(-t / tau1)) +
                beta3 * (factor2 - np.exp(-t / tau2)))

    def fit(
        self,
        tenors: np.ndarray,
        yields: np.ndarray,
        initial_guess: Optional[Tuple] = None
    ) -> None:
        """
        Fit the Nelson-Siegel-Svensson model to yield curve data.

        Parameters
        ----------
        tenors : np.ndarray
            Observed tenors
        yields : np.ndarray
            Observed yields
        initial_guess : tuple, optional
            Initial parameter guess (beta0, beta1, beta2, beta3, tau1, tau2)
            Default: (mean(yields), -0.01, 0.01, 0.01, 2.0, 5.0)
        """
        if initial_guess is None:
            initial_guess = (np.mean(yields), -0.01, 0.01, 0.01, 2.0, 5.0)

        # Vectorized NSS function
        def nss_vec(t, beta0, beta1, beta2, beta3, tau1, tau2):
            return np.array([self._nss(ti, beta0, beta1, beta2, beta3, tau1, tau2)
                           for ti in t])

        # Fit using curve_fit
        try:
            params, _ = curve_fit(
                nss_vec,
                tenors,
                yields,
                p0=initial_guess,
                maxfev=20000
            )
            self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2 = params
        except RuntimeError:
            raise RuntimeError("Nelson-Siegel-Svensson fitting failed to converge")

    def interpolate(self, tenor: float) -> float:
        """
        Interpolate the yield at a given tenor using the fitted NSS model.

        Parameters
        ----------
        tenor : float
            Tenor to interpolate at

        Returns
        -------
        float
            Interpolated yield
        """
        if self.beta0 is None:
            raise ValueError("Interpolator has not been fitted. Call fit() first.")

        return self._nss(tenor, self.beta0, self.beta1, self.beta2,
                        self.beta3, self.tau1, self.tau2)

    def get_parameters(self) -> dict:
        """
        Get the fitted model parameters.

        Returns
        -------
        dict
            Dictionary containing beta0, beta1, beta2, beta3, tau1, and tau2
        """
        return {
            'beta0': self.beta0,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'beta3': self.beta3,
            'tau1': self.tau1,
            'tau2': self.tau2
        }
