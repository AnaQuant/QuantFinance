import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf
import argparse


def analyze_lognormality(ticker, start_date='2020-01-01', end_date=None):
    """
    Analyze whether the log returns of a given stock follow a lognormal distribution.

    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date for historical data in 'YYYY-MM-DD' format
    end_date (str): End date for historical data in 'YYYY-MM-DD' format
    """
    print(f"\n--- Analyzing lognormality of returns for {ticker} ---\n")

    # Download historical data
    print(f"Downloading data for {ticker}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if stock_data.empty:
        print(f"No data found for ticker {ticker}. Please check the symbol.")
        return

    print(f"Downloaded {len(stock_data)} days of data")

    # Calculate log returns
    stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    prices = stock_data['Close_AAPL']
    log_returns = np.log(prices / prices.shift(1)).dropna()
    stock_data['log_returns'] = log_returns
    stock_data = stock_data.dropna()

    # Basic statistics
    mean_return = np.mean(log_returns)
    std_return = np.std(log_returns)
    skewness = stats.skew(log_returns)
    kurtosis = stats.kurtosis(log_returns, fisher=False)  # Use Fisher=False to get the actual kurtosis (not excess)

    print("\n--- Basic Statistics of Log Returns ---")
    print(f"Number of observations: {len(log_returns)}")
    print(f"Mean: {mean_return:.6f}")
    print(f"Standard Deviation: {std_return:.6f}")
    print(f"Skewness: {skewness:.6f} (Normal = 0)")
    print(f"Kurtosis: {kurtosis:.6f} (Normal = 3)")
    print(f"Excess Kurtosis: {kurtosis - 3:.6f}")

    # Normality tests
    jb_stat, jb_pvalue = stats.jarque_bera(log_returns)
    sw_stat, sw_pvalue = stats.shapiro(log_returns)
    ks_stat, ks_pvalue = stats.kstest(log_returns, 'norm', args=(mean_return, std_return))

    print("\n--- Normality Tests ---")
    print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_pvalue:.6f}")
    print(f"Shapiro-Wilk test: statistic={sw_stat:.4f}, p-value={sw_pvalue:.6f}")
    print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.6f}")

    if jb_pvalue < 0.05:
        print("Jarque-Bera test rejects normality at 5% significance level")
    if sw_pvalue < 0.05:
        print("Shapiro-Wilk test rejects normality at 5% significance level")
    if ks_pvalue < 0.05:
        print("Kolmogorov-Smirnov test rejects normality at 5% significance level")

    # Test for ARCH effects (volatility clustering)
    arch_test = het_arch(log_returns)

    print("\n--- ARCH Test for Volatility Clustering ---")
    print(f"LM test statistic: {arch_test[0]:.4f}")
    print(f"LM test p-value: {arch_test[1]:.6f}")
    print(f"F test statistic: {arch_test[2]:.4f}")
    print(f"F test p-value: {arch_test[3]:.6f}")

    if arch_test[1] < 0.05:
        print("ARCH effects detected (rejects constant volatility at 5% significance level)")

    # Create plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Time series of log returns
    plt.subplot(2, 2, 1)
    plt.plot(stock_data.index, log_returns)
    plt.title(f'{ticker} Log Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    plt.grid(True)

    # Plot 2: Histogram with normal distribution overlay
    plt.subplot(2, 2, 2)
    sns.histplot(log_returns, kde=True, stat="density", label="Log Returns")
    x = np.linspace(min(log_returns), max(log_returns), 100)
    plt.plot(x, stats.norm.pdf(x, mean_return, std_return), 'r-', label='Normal Distribution')
    plt.title('Histogram of Log Returns vs Normal Distribution')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # Plot 3: Q-Q plot
    plt.subplot(2, 2, 3)
    qqplot(log_returns, line='s', ax=plt.gca())
    plt.title('Q-Q Plot of Log Returns vs Normal Distribution')
    plt.grid(True)

    # Plot 4: Autocorrelation of returns and squared returns
    plt.subplot(2, 2, 4)
    acf_vals = acf(log_returns, nlags=20, fft=False)  # fft=True is faster for large series
    acf_vals_sqr = acf(log_returns ** 2, nlags=20, fft=False)  # fft=True is faster for large series

    # Convert to Series (optional, but useful for labeling lags)
    acf_returns = pd.Series(acf_vals[1:], index=range(1, 21))  # Skip lag 0
    acf_squared = pd.Series(acf_vals_sqr[1:], index=range(1, 21))  # Skip lag 0
    plt.bar(range(1, 21), acf_returns, width=0.4, label='Log Returns', alpha=0.6)
    plt.bar([x + 0.4 for x in range(1, 21)], acf_squared, width=0.4, label='Squared Log Returns', alpha=0.6)
    plt.axhline(y=1.96 / np.sqrt(len(log_returns)), linestyle='--', color='red', alpha=0.7)
    plt.axhline(y=-1.96 / np.sqrt(len(log_returns)), linestyle='--', color='red', alpha=0.7)
    plt.title('Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{ticker}_lognormality_analysis.png")
    print(f"\nPlot saved as {ticker}_lognormality_analysis.png")
    plt.show()

    # Conclusion
    print("\n--- Conclusion ---")
    if jb_pvalue < 0.05 or sw_pvalue < 0.05 or ks_pvalue < 0.05:
        print("The log returns do NOT follow a normal distribution.")
        if skewness < -0.1 or skewness > 0.1:
            print(f"There is significant skewness ({skewness:.4f}) in the distribution.")
        if kurtosis > 3.5:
            print(f"The distribution has fat tails (excess kurtosis = {kurtosis - 3:.4f}).")
    else:
        print("The log returns appear to follow a normal distribution.")

    if arch_test[1] < 0.05:
        print(
            "There is evidence of volatility clustering, which violates the constant volatility assumption of Black-Scholes.")

    print("\nImplications for Black-Scholes Option Pricing:")
    if jb_pvalue < 0.05 or arch_test[1] < 0.05:
        print("- Basic Black-Scholes model may not be appropriate without adjustments")
        if kurtosis > 3.5:
            print("- The model may underprice out-of-the-money options due to fat tails")
        if arch_test[1] < 0.05:
            print("- Consider models with stochastic volatility (e.g., Heston, SABR)")
        if skewness < -0.2:
            print("- Negative skew suggests using models that account for downside jumps")
    else:
        print("- Basic Black-Scholes assumptions appear reasonable for this equity")

    return stock_data


if __name__ == "__main__":
    # OPTION 1: Command line arguments (uncomment to use)
    """
    parser = argparse.ArgumentParser(description='Analyze stock returns for lognormality')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()
    analyze_lognormality(args.ticker, args.start, args.end)
    """

    # OPTION 2: Direct function call (easier for PyCharm)
    # Change these values as needed
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = '2025-04-30'  # Set to None for current date or specify an end date

    analyze_lognormality(ticker, start_date, end_date)

