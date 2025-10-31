# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a quantitative finance repository containing Python implementations of financial models, stochastic processes, and portfolio optimization techniques. The codebase is organized into modular packages with Jupyter notebooks for analysis and experimentation.

## Project Structure

### Python Packages (Importable Libraries)

- **StochasticProcesses/**: Stochastic process implementations for quantitative finance
  - `base.py`: Abstract base classes for all processes
  - `geometric_brownian_motion.py`: GBM for asset prices
  - `ornstein_uhlenbeck.py`: OU mean-reverting process
  - `cir_process.py`: Cox-Ingersoll-Ross for interest rates
  - `heston_model.py`: Stochastic volatility model
  - `black_derman_toy.py`: Interest rate tree model
  - `utils.py`: Plotting and simulation utilities
  - See `StochasticProcesses/README.md` for detailed documentation

- **FixedIncome/**: Comprehensive fixed income toolkit
  - `bonds.py`: Bond and CallableBond classes with pricing, YTM, duration, convexity (also legacy functions)
  - `yield_curve.py`: YieldCurve class for term structure management
  - `interpolation.py`: CubicSpline, Nelson-Siegel, Nelson-Siegel-Svensson interpolators
  - `data_fetchers.py`: Utilities to fetch Treasury data from web and Excel
  - `pricing.py`: Advanced bond pricing with stochastic interest rate models
  - See `FixedIncome/README.md` for detailed documentation

- **Data/**: Market data utilities
  - `stocks/downloader.py`: yfinance-based downloader for historical stock data (saves to parquet format)

### Jupyter Notebooks (Research and Exploration)

- **Notebooks/**: Interactive analysis and experimentation
  - Stochastic processes: `SP - *.ipynb` files demonstrate GBM, OU, CIR, Heston, BDT models
  - Portfolio optimization: Mean-variance, CVXPY implementations with risk concentration constraints
  - FX strategies: Currency hedging, portfolio exposures, momentum strategies
  - Yield curve: `yield_curve_*.ipynb` for construction and analysis
  - **Note:** Notebooks are kept for learning/experimentation; production code should use the Python packages above

## Code Architecture

### Stochastic Processes Package

**Design Pattern:** Abstract base class with concrete implementations

- **Abstract Base Class** (`StochasticProcess`): Defines interface for all continuous-time processes
  - Required methods: `simulate()`, optional: `expected_value()`, `variance()`
- **Discrete-Time Base** (`DiscreteTimeModel`): For tree-based models (BDT)
- **Concrete Implementations:** Each process extends base class with specific SDE dynamics
- **Utility Module:** Common simulation, plotting, and statistical comparison tools

**Key Principle:** Each process encapsulates its mathematical properties (drift, diffusion, moments) and provides both exact and numerical simulation methods where applicable.

### Fixed Income Architecture

**Design Pattern:** Layered architecture with separation of concerns

**Layer 1 - Data Objects:**
- `Bond` and `CallableBond` classes: Encapsulate bond characteristics and cash flows
- `YieldCurve` class: Manages tenor-yield mappings with date/currency metadata
- `CashFlow` dataclass: Represents individual payment events

**Layer 2 - Analysis Tools:**
- `Interpolator` abstract class with implementations (CubicSpline, Nelson-Siegel, NSS)
- Risk metrics: duration, convexity calculated as Bond methods
- Yield curve shapes: slope, level, curvature computed via YieldCurve methods

**Layer 3 - Data Access:**
- `data_fetchers` module: Fetches Treasury data from web or Excel
- Parser functions: Convert raw data to `YieldCurve` objects

**Layer 4 - Advanced Pricing:**
- `BondPricerWithIRModel`: Integrates Bond objects with StochasticProcesses
- Monte Carlo pricing with CIR paths
- Binomial pricing with BDT trees

**Integration Pattern:** FixedIncome imports from StochasticProcesses for advanced pricing, maintaining clear dependency direction.


### Portfolio Optimization

Uses CVXPY for convex optimization problems with:
- Covariance matrix estimation with exponential weighting
- Risk concentration constraints limiting contribution to portfolio variance
- Sparsity penalties for practical portfolio construction
- Transaction cost modeling

## Development Workflow

### Running Python Modules

Each module can be executed standalone for testing:
```bash
# Stochastic Processes
python StochasticProcesses/geometric_brownian_motion.py
python StochasticProcesses/cir_process.py

# Fixed Income
python FixedIncome/bonds.py

# Options
python OptionPricing/black_scholes_model.py
python OptionPricing/option_pricing.py

# Data
python Data/stocks/downloader.py
```

### Using the Packages in Python Scripts

```python
# Import stochastic processes
from StochasticProcesses import GBM, CIRProcess, HestonModel, BDT
from StochasticProcesses.utils import plot_paths, compare_empirical_theoretical

# Import fixed income tools
from FixedIncome import Bond, CallableBond, YieldCurve
from FixedIncome.interpolation import NelsonSiegelInterpolator
from FixedIncome.data_fetchers import fetch_us_treasury_data
from FixedIncome.pricing import BondPricerWithIRModel

# Example: Price a bond with CIR model
bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10, frequency=2)
cir = CIRProcess(initial_value=0.05, theta=0.5, mu=0.04, sigma=0.02)
pricer = BondPricerWithIRModel(cir)
result = pricer.price(bond, n_paths=10000)
print(f"Bond price: ${result['price']:.2f}")
```

### Working with Notebooks

Start Jupyter from the project root:
```bash
jupyter notebook Notebooks/
```

Import the packages in notebooks:
```python
# At the top of your notebook
import sys
sys.path.append('..')  # Add parent directory to path

from StochasticProcesses import CIRProcess
from FixedIncome import Bond, YieldCurve
```

Notebooks use custom matplotlib stylesheets and require:
- pandas, numpy, scipy
- matplotlib, seaborn
- cvxpy (for portfolio optimization)
- yfinance (for data download)

### Data Management

The `Data/stocks/downloader.py` script:
- Reads ticker symbols from `tickers.txt` (one per line)
- Downloads historical data via yfinance
- Saves to parquet format in `data/` directory
- Implements retry logic for failed downloads

## Module Usage Patterns

### Stochastic Processes

**Pattern 1: Basic Simulation**
```python
from StochasticProcesses import GBM
gbm = GBM(initial_value=100, mu=0.05, sigma=0.2)
t, paths = gbm.simulate(T=1.0, n_steps=252, n_paths=1000)
```

**Pattern 2: Comparing Empirical vs Theoretical**
```python
from StochasticProcesses import CIRProcess
from StochasticProcesses.utils import compare_empirical_theoretical
import numpy as np

cir = CIRProcess(initial_value=0.05, theta=0.5, mu=0.04, sigma=0.02)
t, paths = cir.simulate(T=5.0, n_steps=500, n_paths=10000, method='exact')

# Get theoretical values
theoretical_mean = np.array([cir.expected_value(ti) for ti in t])
theoretical_std = np.array([np.sqrt(cir.variance(ti)) for ti in t])

# Compare
compare_empirical_theoretical(t, paths, theoretical_mean, theoretical_std)
```

**Pattern 3: Using with Fixed Income**
```python
from StochasticProcesses import BDT
from FixedIncome.pricing import BondPricerWithIRModel
from FixedIncome import Bond

# Setup BDT model
bdt = BDT(r0=0.05, volatilities=[0.02]*9, dt=1.0)
bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10)

# Price bond
pricer = BondPricerWithIRModel(bdt)
result = pricer.price(bond)
```

### Fixed Income

**Pattern 1: Object-Oriented Bond Pricing**
```python
from FixedIncome import Bond

# Create and price bond
bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10, frequency=2)
price = bond.price(market_rate=0.04)
ytm = bond.yield_to_maturity(price=1081.11)

# Risk metrics
duration = bond.duration(market_rate=0.04)
convexity = bond.convexity(market_rate=0.04)
```

**Pattern 2: Yield Curve with Interpolation**
```python
from FixedIncome import YieldCurve
from FixedIncome.interpolation import NelsonSiegelInterpolator

# Create curve
yc = YieldCurve(tenors=[0.25, 0.5, 1, 2, 5, 10, 30],
                yields=[0.025, 0.028, 0.032, 0.035, 0.038, 0.040, 0.042])

# Add interpolation
ns = NelsonSiegelInterpolator()
yc.set_interpolator(ns)

# Use curve
yield_7y = yc.get_yield(7.0)
df_5y = yc.get_discount_factor(5.0)
fwd_2_5 = yc.get_forward_rate(2.0, 5.0)
```

**Pattern 3: Market Data to Pricing**
```python
from FixedIncome.data_fetchers import fetch_us_treasury_data, parse_treasury_data_to_yield_curves
from FixedIncome import Bond

# Fetch and parse data
df = fetch_us_treasury_data(year=2025)
curves = parse_treasury_data_to_yield_curves(df)
latest_curve = curves[max(curves.keys())]

# Price bond using curve
bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10, frequency=2)
cash_flows = bond.get_cash_flows()
price = sum(cf.amount * latest_curve.get_discount_factor(cf.time) for cf in cash_flows)
```

## Key Concepts

### Stochastic Process Theory

**Continuous-Time Processes:**
- Defined by stochastic differential equations (SDEs)
- Simulated using Euler-Maruyama or exact methods
- Key properties: drift (μ), diffusion (σ), mean reversion

**Mean Reversion:**
- OU and CIR processes exhibit mean reversion to long-term level μ
- Speed controlled by θ parameter
- Half-life = ln(2)/θ

**Feller Condition:**
- For CIR: 2θμ ≥ σ² ensures process stays positive
- Critical for interest rate and variance modeling

### Fixed Income Theory

**Duration:** Weighted average time to receive cash flows, measures interest rate sensitivity

**Convexity:** Measures curvature of price-yield relationship, second-order risk metric

**Yield Curve Shapes:**
- Normal: upward sloping (long > short rates)
- Inverted: downward sloping (recession indicator)
- Flat: similar rates across maturities
- Humped: mid-term rates highest

**Nelson-Siegel Components:**
- β₀: long-term level
- β₁: short-term/slope
- β₂: curvature/hump
- τ: decay factor

### Black-Scholes Implementation

Black-Scholes functionality is available through:
1. `StochasticProcesses.GBM`: Includes `price_option()` method for Monte Carlo option pricing
2. Inline implementations in notebooks (e.g., "SP - Brownian motion to Black-Scholes.ipynb")

For production option pricing needs, consider creating a dedicated OptionPricing package based on StochasticProcesses.

### Portfolio Optimization with Risk Constraints

The CVXPY notebooks implement variance minimization with:
- Exponentially weighted covariance matrices (decay parameter ~0.97)
- Position limits (w_min, w_max bounds)
- Risk concentration constraints: `x^T Σ_i x ≤ rcr * λ_i * x^T Σ x` where Σ_i is the i-th component covariance and λ_i is eigenvalue
- L1 norm penalties for sparsity

### Yield Curve Construction

Notebooks demonstrate:
- Fetching Treasury data from treasury.gov
- Cubic spline interpolation for missing maturities
- Visualization of term structure evolution

## Common Patterns

### Reading Financial Data

Most notebooks use:
```python
df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
```

### Covariance Matrix Calculations

Exponentially weighted covariance for time-varying risk:
```python
decay = 0.97
cov_matrix = returns.ewm(com=decay/(1.0-decay), adjust=False).cov(bias=True)
```

### Adjusted Returns

Momentum-adjusted returns for portfolio optimization:
```python
spot_momentum = spot_return.ewm(com=decay/(1.0-decay), adjust=False).mean()
adj_return = spot_return - spot_momentum
```

## Virtual Environment

The repository uses Python virtual environments (`.venv` or `.venv-1`). Activate before working:
```bash
source .venv/bin/activate  # or .venv-1/bin/activate
```

## Git Workflow

Recent commits show active development on:
- Black-Scholes model class implementation
- Data pipeline improvements
- Codebase cleanup and refactoring

When committing, focus on modular changes to individual packages rather than broad changes across notebooks.
