import QuantLib as ql
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import brentq
from scipy.interpolate import griddata


class VolatilitySurface:
    def __init__(self, spot_price, risk_free_rate, volatility, use_black_scholes=False):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.use_black_scholes = use_black_scholes
        self.calendar = ql.NullCalendar()
        self.day_count = ql.Actual365Fixed()
        self.spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
        self.rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, self.calendar,
                                                                      ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)),
                                                                      self.day_count))
        self.vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, self.calendar,
                                                                             ql.QuoteHandle(ql.SimpleQuote(volatility)),
                                                                             self.day_count))
        self.bs_process = ql.BlackScholesProcess(self.spot_handle, self.rate_handle, self.vol_handle)
        self.option = None

    def svi_total_variance(self, params, k):
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def svi_objective(self, params, k, market_variance):
        model_variance = self.svi_total_variance(params, k)
        return np.sum((market_variance - model_variance) ** 2)  # Minimize MSE

    def calibrate_svi(self, strikes, maturities, market_vols, F):
        log_moneyness = np.log(strikes / F)
        market_variance = (market_vols ** 2) * maturities[:, np.newaxis]

        svi_params = []
        for j, T in enumerate(maturities):
            initial_guess = [0.1, 0.1, 0.0, 0.0, 0.1]
            bounds = [(0, None), (0, None), (-1, 1), (-np.inf, np.inf), (0, None)]
            result = opt.minimize(self.svi_objective, initial_guess, args=(log_moneyness, market_variance[j, :]),
                                  bounds=bounds, method='L-BFGS-B')
            svi_params.append(result.x)
        return np.array(svi_params)


    def black_scholes_call(self, spot, strike, term, int_rate, sigma):
        d1 = (log(spot / strike) + (int_rate + 0.5 * sigma ** 2) * term) / (sigma * sqrt(term))
        d2 = d1 - sigma * sqrt(term)
        call_price = spot * norm.cdf(d1) - strike * exp(-int_rate * term) * norm.cdf(d2)
        return call_price


    # def implied_volatility(self, market_price, spot, strike, term, int_rate):
    #     objective = lambda sigma: self.black_scholes_call(spot, strike, term, int_rate, sigma) - market_price
    #     return brentq(objective, 0.01, 5.0)  # Bounds for volatility


    def generate_synthetic_prices(self, strikes, maturities):

        prices = np.zeros((len(strikes), len(maturities)))

        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                if self.use_black_scholes:
                    prices[i, j] = self.black_scholes_call(self.spot_price, K, T, self.risk_free_rate, self.volatility)
                else:
                    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                                              ql.EuropeanExercise(ql.Date.todaysDate() + ql.Period(int(T*365), ql.Days)))
                    engine = ql.AnalyticEuropeanEngine(self.bs_process)
                    option.setPricingEngine(engine)
                    prices[i, j] = option.NPV()
                    self.option = option


        return prices

    def compute_implied_vols(self, strikes, maturities, prices):
        """
        Compute implied volatilities for a given set of strikes, maturities, and option prices.
        Uses QuantLib's impliedVolatility() function.
        """
        # Container for implied volatilities
        implied_vols = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                try:
                    implied_vol = self.option.impliedVolatility(prices[i, j], self.bs_process)
                    implied_vols[i, j] = implied_vol
                except RuntimeError:
                    implied_vols[i, j] = np.nan  # If no solution is found, assign NaN

        return implied_vols



    def plot_vol_surface(self, strikes, maturities, imp_vols):
        X, Y = np.meshgrid(strikes, maturities)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, imp_vols.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity (Years)')
        ax.set_zlabel('Option Price')
        # ax.set_title('Synthetic Option Prices Volatility Surface')
        ax.set_title('Synthetic Implied Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plot_vol_surface_svi(self, strikes, maturities, svi_params, F):
        log_moneyness = np.log(strikes / F)
        svi_vol_surface = np.array([[np.sqrt(self.svi_total_variance(svi_params[j], k) / T)
                                     for k in log_moneyness] for j, T in enumerate(maturities)])

        X, Y = np.meshgrid(strikes, maturities)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, svi_vol_surface.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity (Years)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('SVI Calibrated Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == "__main__":
    spot_price = 100
    risk_free_rate = 0.02
    volatility = 0.2
    use_black_scholes = False  # Set to False to use QuantLib pricing
    # Simulation data
    strikes = np.linspace(80, 120, 10)
    maturities = np.linspace(0.1, 2, 10)

    vol_surface = VolatilitySurface(spot_price, risk_free_rate, volatility, use_black_scholes)
    prices = vol_surface.generate_synthetic_prices(strikes, maturities)
    imp_vols = vol_surface.compute_implied_vols(strikes, maturities, prices)
    # market_vols = np.random.rand(len(maturities), len(strikes)) * 0.2 + 0.1
    svi_params = vol_surface.calibrate_svi(strikes, maturities, imp_vols, F=spot_price)
    # vol_surface.plot_vol_surface(strikes, maturities, imp_vols)
    vol_surface.plot_vol_surface_svi(strikes, maturities, svi_params, F=spot_price)
