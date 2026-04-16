# QuantFinance — Fixed Income Analytics

A practitioner-level fixed income analytics framework for bank treasury and ALM contexts. Covers the full stack from yield curve foundations to carry decomposition, OCI risk, and Monte Carlo capital simulation.

---

## Notebook Series

The notebooks are designed as a progressive series. Each layer assumes the one before it.

### Layer 0 — Yield Curves: Theory and Practice [`Yield_Curve_Theory_and_Practice.ipynb`](Notebooks/Yield_Curve_Theory_and_Practice.ipynb)
Yield curve construction from live US Treasury data, three interpolation methods (cubic spline, Nelson-Siegel, NSS), spot/forward/discount factor decomposition, bond pricing off the curve, duration and convexity, scenario analysis, and trading strategies (roll-down, bullet vs barbell). Uses the `YieldCurve`, `Bond`, and interpolation classes from the `FixedIncome` package.

### Layer 1 — Carry, OCI, and Monte Carlo CET1 [`fixed_income_treasury_analytics.ipynb`](Notebooks/fixed_income_treasury_analytics.ipynb)
Carry decomposition across a three-bond UK gilt portfolio, a deterministic OCI scenario tool (IFRS 9 / CET1 impact of a +100 bps parallel shift), and a Monte Carlo framework producing a full CET1 distribution with tail statistics. An interactive Streamlit version of the OCI tool is in [`Apps/oci_tool.py`](Apps/oci_tool.py).

### Layer 2 — Portfolio Construction and DV01 Hedging [`quant_prep_treasury.ipynb`](Notebooks/quant_prep_treasury.ipynb)
Portfolio representation and DV01 computation for a synthetic UK bank treasury book (gilts + ESHLA loans). Builds to a CVXPY optimisation with regulatory constraints (Basel III RWA, LCR) and an internal DV01 limit, showing which constraints bind and why.

### Layer 3 — Convexity, ALM Constraints, and Regime Detection [`fixed_income_convexity_alm.ipynb`](Notebooks/fixed_income_convexity_alm.ipynb)
Convexity as a second-order correction to DV01-based P&L; the NII vs OCI trade-off as the binding ALM constraint in a banking book; and a Hidden Markov Model applied to interest rate regime detection.

---

## Interactive App

**OCI Scenario Tool** ([`Apps/oci_tool.py`](Apps/oci_tool.py)) — Streamlit companion to Layer 1. Applies a parallel rate shock to the AFS gilt portfolio and computes the mark-to-market OCI loss and CET1 impact interactively.

```bash
streamlit run Apps/oci_tool.py
```

---

## Structure

```
QuantFinance/
├── Apps/
│   └── oci_tool.py                          # Streamlit OCI scenario tool
├── FixedIncome/
│   ├── bonds.py                             # Bond pricing, DV01, duration, convexity
│   ├── yield_curve.py                       # YieldCurve class
│   ├── interpolation.py                     # Cubic spline, Nelson-Siegel, NSS
│   ├── data_fetchers.py                     # US Treasury data fetchers
│   └── README.md                            # Package API reference
├── Notebooks/
│   ├── Yield_Curve_Theory_and_Practice.ipynb
│   ├── fixed_income_treasury_analytics.ipynb
│   ├── quant_prep_treasury.ipynb
│   ├── fixed_income_convexity_alm.ipynb
│   └── README.md                            # Notebook descriptions
├── requirements.txt
└── README.md
```

---

## Dependencies

```
numpy  pandas  scipy  matplotlib  cvxpy  hmmlearn  streamlit
```

```bash
pip install -r requirements.txt
```

---

## DV01 Conventions

Layer 2 computes portfolio-level DV01 in £m per basis point using `notional × modified_duration × 0.0001`. Layers 1–3 use bump-and-reprice via `compute_dv01`, returning £ per £100 face per basis point. Both are standard approaches; the difference reflects portfolio-level vs. instrument-level analysis.
