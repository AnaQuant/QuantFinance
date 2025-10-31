# Stochastic Processes Package

This package provides implementations of various stochastic processes used in quantitative finance.

## Overview

Stochastic processes are mathematical models for quantities that evolve randomly over time. In quantitative finance, they are fundamental for modeling asset prices, interest rates, and volatility.

## Available Processes

### 1. Geometric Brownian Motion (GBM)
**File:** `geometric_brownian_motion.py`

The standard model for asset prices in the Black-Scholes framework.

**SDE:** `dS(t) = μ S(t) dt + σ S(t) dW(t)`

**Key Features:**
- Exact solution available
- Euler-Maruyama discretization also provided
- European option pricing via Monte Carlo
- Theoretical moments: `E[S(t)] = S₀ exp(μt)`

**Example:**
```python
from StochasticProcesses import GBM

gbm = GBM(initial_value=100, mu=0.05, sigma=0.2)
t, paths = gbm.simulate(T=1.0, n_steps=252, n_paths=1000)

# Price an option
result = gbm.price_option(K=100, T=1.0, r=0.05, option_type='call')
print(f"Call price: {result['price']:.4f}")
```

### 2. Ornstein-Uhlenbeck (OU) Process
**File:** `ornstein_uhlenbeck.py`

A mean-reverting process commonly used for interest rates and spreads.

**SDE:** `dX(t) = θ(μ - X(t))dt + σ dW(t)`

**Key Features:**
- Mean reversion to long-term level μ
- Speed of reversion controlled by θ
- Both Euler and exact simulation methods
- Stationary distribution: `N(μ, σ²/(2θ))`
- Half-life calculation

**Example:**
```python
from StochasticProcesses import OU

ou = OU(initial_value=0.05, theta=0.5, mu=0.03, sigma=0.01)
t, paths = ou.simulate_exact(T=5.0, n_steps=500, n_paths=1000)

print(f"Half-life: {ou.half_life():.2f} years")
```

### 3. Cox-Ingersoll-Ross (CIR) Process
**File:** `cir_process.py`

A mean-reverting process that ensures non-negative values, ideal for interest rates.

**SDE:** `dX(t) = θ(μ - X(t))dt + σ√X(t) dW(t)`

**Key Features:**
- Guaranteed non-negativity (under Feller condition: 2θμ ≥ σ²)
- Mean reversion with state-dependent volatility
- Exact simulation via non-central chi-squared distribution
- Marginal distribution available

**Example:**
```python
from StochasticProcesses import CIRProcess

cir = CIRProcess(initial_value=0.05, theta=0.5, mu=0.03, sigma=0.05)

# Check Feller condition
if cir.satisfies_feller_condition():
    print("Process will remain positive")

# Simulate
t, paths = cir.simulate(T=5.0, n_steps=500, n_paths=1000, method='exact')
```

### 4. Heston Stochastic Volatility Model
**File:** `heston_model.py`

A two-factor model where both price and volatility are stochastic.

**SDEs:**
```
dS(t) = μ S(t) dt + √v(t) S(t) dW₁(t)
dv(t) = κ(θ - v(t))dt + σᵥ√v(t) dW₂(t)
```
with correlation ρ between W₁ and W₂.

**Key Features:**
- Captures volatility clustering and mean reversion
- Correlated asset and volatility processes
- Euler-Maruyama and Milstein discretizations
- Separate volatility and variance path extraction

**Example:**
```python
from StochasticProcesses import HestonModel

heston = HestonModel(
    initial_value=100,
    initial_variance=0.04,
    mu=0.05,
    kappa=2.0,
    theta=0.04,
    sigma_v=0.3,
    rho=-0.7
)

t, price_paths, variance_paths = heston.simulate(T=1.0, n_steps=252, n_paths=1000)

# Get volatility paths
t, vol_paths = heston.get_volatility_paths(T=1.0, n_steps=252, n_paths=1000)
```

### 5. Black-Derman-Toy (BDT) Model
**File:** `black_derman_toy.py`

A discrete-time binomial tree model for interest rates.

**Dynamics:** `r_{up} = r * exp(2σ√Δt)`, `r_{down} = r * exp(-2σ√Δt)`

**Key Features:**
- No-arbitrage binomial tree construction
- Calibrated to market yield curve and volatilities
- Bond pricing via backward induction
- Both zero-coupon and coupon-bearing bonds

**Example:**
```python
from StochasticProcesses import BDT

# Time-dependent volatilities
volatilities = [0.02, 0.021, 0.022, 0.023, 0.024]

bdt = BDT(r0=0.05, volatilities=volatilities, dt=1.0)
tree = bdt.build_tree()

# Price a zero-coupon bond
price = bdt.price_zero_coupon_bond(face_value=100)
print(f"Zero-coupon bond price: {price:.4f}")

# Price a coupon bond
coupon_price = bdt.price_coupon_bond(face_value=100, coupon_rate=0.05)
print(f"Coupon bond price: {coupon_price:.4f}")
```

## Utility Functions

**File:** `utils.py`

Common utilities for all processes:
- `plot_paths()`: Visualize simulated paths
- `compare_empirical_theoretical()`: Compare simulation results with theory
- `plot_distribution()`: Plot terminal distribution
- `generate_brownian_increments()`: Generate Brownian motion

## Mathematical Background

### Stochastic Differential Equations (SDEs)

All continuous-time processes satisfy an SDE of the form:
```
dX(t) = μ(X,t)dt + σ(X,t)dW(t)
```

where:
- μ(X,t) is the drift term
- σ(X,t) is the diffusion term
- W(t) is a Brownian motion

### Numerical Methods

1. **Euler-Maruyama Scheme:**
   ```
   X_{n+1} = X_n + μ(X_n,t_n)Δt + σ(X_n,t_n)ΔW_n
   ```

2. **Milstein Scheme** (higher order):
   ```
   X_{n+1} = X_n + μ(X_n,t_n)Δt + σ(X_n,t_n)ΔW_n +
             (σ∂σ/∂x/2)(ΔW_n² - Δt)
   ```

3. **Exact Simulation:** When analytical solutions exist (GBM, OU, CIR*)

## Usage Patterns

### Basic Simulation
```python
from StochasticProcesses import GBM, CIRProcess
from StochasticProcesses.utils import plot_paths

# Simulate GBM
gbm = GBM(initial_value=100, mu=0.05, sigma=0.2)
t, paths = gbm.simulate(T=1.0, n_steps=252, n_paths=1000)

# Plot results
plot_paths(t, paths, title="GBM Simulation", show_mean=True)
```

### Comparing Models
```python
from StochasticProcesses import OU, CIRProcess
import numpy as np

# Same parameters for comparison
theta, mu, sigma = 0.5, 0.03, 0.01

ou = OU(initial_value=0.05, theta=theta, mu=mu, sigma=sigma)
cir = CIRProcess(initial_value=0.05, theta=theta, mu=mu, sigma=sigma)

t_ou, paths_ou = ou.simulate(T=5.0, n_steps=500, n_paths=1000)
t_cir, paths_cir = cir.simulate(T=5.0, n_steps=500, n_paths=1000)

# Compare mean paths
print(f"OU final mean: {np.mean(paths_ou[:, -1]):.4f}")
print(f"CIR final mean: {np.mean(paths_cir[:, -1]):.4f}")
```

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities
- Cox, J. C., Ingersoll, J. E., & Ross, S. A. (1985). A Theory of the Term Structure of Interest Rates
- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility
- Black, F., Derman, E., & Toy, W. (1990). A One-Factor Model of Interest Rates
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering
