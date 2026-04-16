# Fixed Income Package

This package provides comprehensive tools for fixed income analysis, including bond pricing, yield curve management, and advanced interest rate modeling.

## Modules Overview

### 1. bonds.py - Bond Pricing and Analysis

Provides both object-oriented and functional interfaces for bond valuation.

#### Object-Oriented Interface (Recommended)

**Bond Class:**
```python
from FixedIncome import Bond

bond = Bond(
    face_value=1000,
    coupon_rate=0.05,  # 5% annual coupon
    maturity=10,       # 10 years
    frequency=2        # Semi-annual payments
)

# Price the bond
price = bond.price(market_rate=0.04)
print(f"Bond price: ${price:.2f}")

# Calculate yield to maturity
ytm = bond.yield_to_maturity(price=1081.11)
print(f"YTM: {ytm:.4%}")

# Risk metrics
duration = bond.duration(market_rate=0.04)
modified_duration = bond.modified_duration(market_rate=0.04)
convexity = bond.convexity(market_rate=0.04)

print(f"Macaulay Duration: {duration:.4f} years")
print(f"Modified Duration: {modified_duration:.4f}")
print(f"Convexity: {convexity:.4f}")
```

**CallableBond Class:**
```python
from FixedIncome import CallableBond

callable_bond = CallableBond(
    face_value=1000,
    coupon_rate=0.06,
    maturity=10,
    call_price=1050,
    call_date=5,
    frequency=2
)

price = callable_bond.price(market_rate=0.04)
ytc = callable_bond.yield_to_call(price=price)
print(f"Yield to Call: {ytc:.4%}")
```

#### Functional Interface (Legacy)

Original functions are still available for backwards compatibility:
- `bond_price()`
- `yield_to_maturity()`
- `callable_bond_price()`
- `bond_duration()`

### 2. yield_curve.py - Yield Curve Management

The `YieldCurve` class manages and analyzes interest rate term structures.

```python
from FixedIncome import YieldCurve
from FixedIncome.interpolation import CubicSplineInterpolator
from datetime import datetime

# Create a yield curve
tenors = [0.25, 0.5, 1, 2, 5, 10, 30]
yields = [0.025, 0.028, 0.032, 0.035, 0.038, 0.040, 0.042]

yc = YieldCurve(
    tenors=tenors,
    yields=yields,
    date=datetime(2025, 1, 1),
    currency='USD'
)

# Add interpolation
interpolator = CubicSplineInterpolator()
yc.set_interpolator(interpolator)

# Get interpolated yields
yield_7y = yc.get_yield(7.0)
print(f"7-year yield: {yield_7y:.4%}")

# Calculate discount factors
df_5y = yc.get_discount_factor(5.0)
print(f"5-year discount factor: {df_5y:.6f}")

# Get forward rates
forward_2_5 = yc.get_forward_rate(2.0, 5.0)
print(f"2y-5y forward rate: {forward_2_5:.4%}")

# Analyze curve shape
metrics = yc.get_shape_metrics()
print(f"2-10 slope: {metrics['slope_2_10']:.4%}")
print(f"Curvature: {metrics['curvature']:.4%}")

# Plot the curve
yc.plot()
```

### 3. interpolation.py - Yield Curve Interpolation

Three interpolation methods are provided:

#### Cubic Spline Interpolation
```python
from FixedIncome.interpolation import CubicSplineInterpolator

interpolator = CubicSplineInterpolator(bc_type='natural')
interpolator.fit(tenors, yields)

# Interpolate at any tenor
yield_3_5y = interpolator.interpolate(3.5)
```

#### Nelson-Siegel Model
```python
from FixedIncome.interpolation import NelsonSiegelInterpolator

ns = NelsonSiegelInterpolator()
ns.fit(tenors, yields)

# Get parameters
params = ns.get_parameters()
print(f"Long-term level (β₀): {params['beta0']:.4%}")
print(f"Short-term component (β₁): {params['beta1']:.4%}")
print(f"Curvature (β₂): {params['beta2']:.4%}")
print(f"Decay factor (τ): {params['tau']:.4f}")

# Interpolate
yield_8y = ns.interpolate(8.0)
```

#### Nelson-Siegel-Svensson Model
```python
from FixedIncome.interpolation import NelsonSiegelSvenssonInterpolator

nss = NelsonSiegelSvenssonInterpolator()
nss.fit(tenors, yields)

# More flexible for complex curve shapes
params = nss.get_parameters()
```

### 4. data_fetchers.py - Market Data Retrieval

Utilities for fetching yield curve data from various sources.

#### US Treasury Data
```python
from FixedIncome.data_fetchers import (
    fetch_us_treasury_data,
    parse_treasury_data_to_yield_curves
)

# Fetch current year data
df = fetch_us_treasury_data(year=2025)

# Parse into YieldCurve objects
curves = parse_treasury_data_to_yield_curves(df)

# Get latest curve
latest_date = max(curves.keys())
latest_curve = curves[latest_date]
print(f"Latest curve date: {latest_date}")
latest_curve.plot()
```

#### Historical Data
```python
from FixedIncome.data_fetchers import fetch_historical_treasury_data

# Fetch multiple years
df_hist = fetch_historical_treasury_data(start_year=2020, end_year=2025)
print(f"Downloaded {len(df_hist)} observations")
```

#### Excel Data
```python
from FixedIncome.data_fetchers import (
    load_yield_curve_from_excel,
    create_yield_curve_from_excel_row
)

# Load from Excel
data = load_yield_curve_from_excel('yield_curves.xlsx')

# Create curve from specific row
usd_data = data['USD_Z0']
curve = create_yield_curve_from_excel_row(usd_data, row_index=-1, currency='USD')
```

## Complete Workflow Example

```python
from FixedIncome import Bond, YieldCurve
from FixedIncome.data_fetchers import fetch_us_treasury_data, parse_treasury_data_to_yield_curves
from FixedIncome.interpolation import NelsonSiegelInterpolator

# 1. Fetch market data and build curve
df = fetch_us_treasury_data(year=2025)
curves = parse_treasury_data_to_yield_curves(df)
yc = curves[max(curves.keys())]

# 2. Attach interpolator
yc.set_interpolator(NelsonSiegelInterpolator())

# 3. Create a bond and price off the curve
bond = Bond(face_value=1000, coupon_rate=0.05, maturity=10, frequency=2)
cash_flows = bond.get_cash_flows()
price = sum(cf.amount * yc.get_discount_factor(cf.time) for cf in cash_flows)
print(f"Price (yield curve): ${price:.2f}")

# 4. Risk metrics
flat_rate = yc.get_yield(10.0)
print(f"Duration:   {bond.duration(flat_rate):.4f} years")
print(f"Convexity:  {bond.convexity(flat_rate):.4f}")

# 5. Curve shape
metrics = yc.get_shape_metrics()
print(f"Level: {metrics['level']:.4%}  Slope (2-10): {metrics['slope_2_10']:.4%}")
```

## Key Concepts

### Duration and Convexity

**Macaulay Duration:** Weighted average time to receive cash flows
```
D = Σ(t * PV(CF_t)) / Price
```

**Modified Duration:** First-order price sensitivity
```
MD = D / (1 + y/m)
ΔP/P ≈ -MD * Δy
```

**Convexity:** Second-order price sensitivity
```
C = Σ(t*(t+1) * PV(CF_t)) / (Price * (1+y/m)²)
ΔP/P ≈ -MD * Δy + 0.5 * C * (Δy)²
```

### Yield Curve Shapes

- **Normal (Upward Sloping):** Long rates > short rates
- **Inverted:** Short rates > long rates (recession indicator)
- **Flat:** Similar rates across maturities
- **Humped:** Mid-term rates highest

### Nelson-Siegel Interpretation

- **β₀:** Long-term level (as t → ∞)
- **β₁:** Short-term component (slope at t=0)
- **β₂:** Medium-term component (curvature)
- **τ:** Decay factor (determines location of hump/trough)

## References

- Hull, J. C. (2017). Options, Futures, and Other Derivatives
- Brigo, D., & Mercurio, F. (2006). Interest Rate Models - Theory and Practice
- Svensson, L. E. O. (1994). Estimating and Interpreting Forward Interest Rates
