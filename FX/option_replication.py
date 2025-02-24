import numpy as np
import pandas as pd

from scipy.stats import norm

# The spot given in the csv is the baseFX
def log_return(spot):

    return np.log(spot).diff()



def realisedVol(spot_return, decay):
    returnEWMA = pd.ewma(spot_return, com=decay / (1.0 - decay), adjust=False)
    adjReturn = spot_return - returnEWMA
    adjReturnSqr = 252 * adjReturn * adjReturn
    varianceEWMA = pd.ewma(adjReturnSqr, com=decay / (1.0 - decay), adjust=False)
    realVol = np.sqrt(varianceEWMA)
    return realVol


def modelVol(realVol, imp_vol):
    vol = np.maximum(realVol, imp_vol / 100)
    return vol


def tStat_deltaRR(RR, decay):
    deltaRR = RR.diff()
    deltaRRsqr = deltaRR * deltaRR
    beta = pd.ewma(deltaRR, com=decay / (1.0 - decay), adjust=False, min_periods=200)
    gamma = pd.ewma(deltaRRsqr, com=decay / (1.0 - decay), adjust=False, min_periods=200)
    window_size = round(1.5 * (1 + decay) / (1 - decay))
    stdErr = np.sqrt((gamma - beta * beta) / (window_size - 1))
    result = beta / stdErr
    return result


def smoothingTstats(e):
    smt_param = 1
    if np.abs(e) > smt_param:
        return np.sign(e)
    else:
        return np.sin(e * np.pi / (2 * smt_param))


def commonTerm(IRb, IRf, vol, T):
    res = (IRb - IRf) * T / 100 + vol * vol * T / 2
    return res


def calcHraw(spot, IR1, IR2, RR, imp_vol, decay, deltaShift, deltaATM):
    # Realised volatility:
    returnBaseCur = log_return(spot)
    realVol = realisedVol(returnBaseCur, decay)
    vol = modelVol(realVol, imp_vol)

    # Risk reversal stuff:
    tstats = tStat_deltaRR(RR, decay)
    smtTstats = tstats.apply(smoothingTstats, args=())
    Delta4Ks = smtTstats * deltaShift + deltaATM
    d1 = norm.ppf(Delta4Ks)

    T = 4 / 52
    hraw = 0
    comTerm = commonTerm(IR1, IR2, vol, T)

    for i in range(1, 6):
        strike = spot.shift(i) / np.exp(np.sqrt(T) * d1 * vol.shift(i) - comTerm.shift(i))
        delta = norm.cdf((np.log(spot / strike) + comTerm) / (vol * np.sqrt(T)))
        hraw = hraw + delta

    return hraw / 5.0