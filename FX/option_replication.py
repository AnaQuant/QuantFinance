import numpy as np


def smoothing_tstats(e):
    smt_param = 1
    if np.abs(e) > smt_param:
        return np.sign(e)
    else:
        return np.sin(e * np.pi / (2 * smt_param))

class RiskReversal:
    def __init__(self, spot, IR1, IR2, RR, imp_vol, decay, deltaShift, deltaATM):
        self.spot = spot
        self.IR1 = IR1
        self.IR2 = IR2
        self.RR = RR
        self.imp_vol = imp_vol
        self.decay = decay
        self.deltaShift = deltaShift
        self.deltaATM = deltaATM

    # The spot given in the csv is the baseFX
    def log_return(self, spot):

        return np.log(spot).diff()

    def realised_vol(self, spot_return):
        return_ewma = spot_return.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()
        adjReturn = spot_return - return_ewma
        adjReturnSqr = 252 * adjReturn * adjReturn
        variance_ewma = adjReturnSqr.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()
        return np.sqrt(variance_ewma)

    def modelVol(self, realVol, imp_vol):
        vol = np.maximum(realVol, imp_vol / 100)
        return vol

    def tstat_deltaRR(self, RR):
        deltaRR = RR.diff()
        deltaRRsqr = deltaRR * deltaRR
        beta = deltaRR.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()
        gamma = deltaRRsqr.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()
        window_size = round(1.5 * (1 + self.decay) / (1 - self.decay))
        stdErr = np.sqrt((gamma - beta * beta) / (window_size - 1))
        result = beta / stdErr

        return result

    def get_data(self):
        pass

    def run_strategy(self):
        pass

    def plot_results(self):
        pass




    # def commonTerm(IRb, IRf, vol, T):
    #     res = (IRb - IRf) * T / 100 + vol * vol * T / 2
    #     return res
    #
    # def calcHraw(spot, IR1, IR2, RR, imp_vol, decay, deltaShift, deltaATM):
    #     # Realised volatility:
    #     returnBaseCur = log_return(spot)
    #     realVol = realised_vol(returnBaseCur, decay)
    #     vol = modelVol(realVol, imp_vol)
    #
    #     # Risk reversal stuff:
    #     tstats = tstat_deltaRR(RR, decay)
    #     smtTstats = tstats.apply(smoothing_tstats, args=())
    #     Delta4Ks = smtTstats * deltaShift + deltaATM
    #     d1 = norm.ppf(Delta4Ks)
    #
    #     T = 4 / 52
    #     hraw = 0
    #     comTerm = commonTerm(IR1, IR2, vol, T)
    #
    #     for i in range(1, 6):
    #         strike = spot.shift(i) / np.exp(np.sqrt(T) * d1 * vol.shift(i) - comTerm.shift(i))
    #         delta = norm.cdf((np.log(spot / strike) + comTerm) / (vol * np.sqrt(T)))
    #         hraw = hraw + delta
    #
    #     return hraw / 5.0