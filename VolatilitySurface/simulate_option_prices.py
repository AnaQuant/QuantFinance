import QuantLib as ql
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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

    def black_scholes_call(self, spot, strike, term, int_rate, sigma):
        d1 = (log(spot / strike) + (int_rate + 0.5 * sigma ** 2) * term) / (sigma * sqrt(term))
        d2 = d1 - sigma * sqrt(term)
        call_price = spot * norm.cdf(d1) - strike * exp(-int_rate * term) * norm.cdf(d2)
        return call_price

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

        return prices

    def plot_vol_surface(self, strikes, maturities, prices):
        X, Y = np.meshgrid(strikes, maturities)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, prices.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity (Years)')
        ax.set_zlabel('Option Price')
        ax.set_title('Synthetic Option Prices Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == "__main__":
    spot_price = 100
    risk_free_rate = 0.03
    volatility = 0.25
    use_black_scholes = False  # Set to False to use QuantLib pricing

    strikes = np.linspace(70, 150, 30)
    maturities = np.linspace(0.1, 5, 20)

    vol_surface = VolatilitySurface(spot_price, risk_free_rate, volatility, use_black_scholes)
    prices = vol_surface.generate_synthetic_prices(strikes, maturities)
    vol_surface.plot_vol_surface(strikes, maturities, prices)
