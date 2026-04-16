# Fixed Income Analytics — Notebook Series

Four notebooks covering yield curve theory, carry analytics, portfolio optimisation, and regime detection. Each layer builds on the one before it.

---

### Layer 0 — `Yield_Curve_Theory_and_Practice.ipynb`
Yield curve construction from live US Treasury data. Covers: cubic spline, Nelson-Siegel, and NSS interpolation; spot rates, forward rates, and discount factors; bond pricing off the curve; duration, convexity, and DV01; scenario analysis (parallel, steepening, flattening); roll-down and bullet vs barbell strategies. *Prerequisite for all other notebooks.*

### Layer 1 — `fixed_income_treasury_analytics.ipynb`
Carry decomposition (coupon income, roll-down, pull-to-par, funding cost) across a three-bond UK gilt portfolio. Deterministic OCI scenario tool: IFRS 9 / CET1 impact of a +100 bps parallel shift. Monte Carlo framework (1,000 simulations) producing a full CET1 distribution with 5th-percentile and breach probability. Interactive Streamlit companion: `../Apps/oci_tool.py`.

### Layer 2 — `quant_prep_treasury.ipynb`
Portfolio representation and DV01 computation for a synthetic UK bank treasury book (gilts + ESHLA loans). CVXPY optimisation with regulatory constraints (Basel III RWA, LCR) and an internal DV01 limit — shows which constraints bind and why.

### Layer 3 — `fixed_income_convexity_alm.ipynb`
Convexity as a second-order correction to DV01-based P&L. NII vs OCI trade-off as the binding ALM constraint in a banking book. Hidden Markov Model for interest rate regime detection.
