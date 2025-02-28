import numpy as np
import scipy.stats as stats

def historical_var(returns, confidence_level=0.95):
    """
    Compute Historical VaR.
    :param returns: (numpy array or list) Historical returns
    :param confidence_level: (float) Confidence level (e.g., 0.95 for 95% VaR)
    :return: (float) VaR estimate
    """
    return -np.percentile(returns, 100 * (1 - confidence_level))

def parametric_var(mean, std_dev, confidence_level=0.95):
    """
    Compute Parametric VaR (Variance-Covariance method, assuming normality).
    :param mean: (float) Expected return
    :param std_dev: (float) Standard deviation of returns
    :param confidence_level: (float) Confidence level (e.g., 0.95 for 95% VaR)
    :return: (float) VaR estimate
    """
    z_score = stats.norm.ppf(1 - confidence_level)
    return -(mean + z_score * std_dev)

def monte_carlo_var(returns, simulations=10000, confidence_level=0.95):
    """
    Compute Monte Carlo VaR by simulating returns.
    :param returns: (numpy array) Historical returns
    :param simulations: (int) Number of simulations
    :param confidence_level: (float) Confidence level (e.g., 0.95 for 95% VaR)
    :return: (float) VaR estimate
    """
    mean, std_dev = np.mean(returns), np.std(returns)
    simulated_returns = np.random.normal(mean, std_dev, simulations)
    return -np.percentile(simulated_returns, 100 * (1 - confidence_level))

def cornish_fisher_var(mean, std_dev, skewness, kurtosis, confidence_level=0.95):
    """
    Compute Cornish-Fisher VaR to account for skewness and kurtosis.
    :param mean: (float) Expected return
    :param std_dev: (float) Standard deviation of returns
    :param skewness: (float) Skewness of returns
    :param kurtosis: (float) Excess kurtosis of returns
    :param confidence_level: (float) Confidence level (e.g., 0.95 for 95% VaR)
    :return: (float) Adjusted VaR estimate
    """
    z = stats.norm.ppf(1 - confidence_level)
    z_cf = (z + (1/6)*(z**2 - 1)*skewness + (1/24)*(z**3 - 3*z)*kurtosis - (1/36)*(2*z**3 - 5*z)*skewness**2)
    return -(mean + z_cf * std_dev)
