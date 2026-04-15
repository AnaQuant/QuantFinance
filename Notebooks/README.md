# Fixed Income Analytics — Notebook Series

Three notebooks documenting a systematic FX researcher's exploration of fixed income analytics for bank treasury and ALM contexts. Written at practitioner level; assumes familiarity with bond pricing, duration, and time-series modelling.

---

## Notebooks

### Day 1 — `quant_prep_treasury.ipynb`
Portfolio representation, DV01 computation, and payer swap hedging for a synthetic UK bank treasury book (gilts + ESHLA). Builds to a CVXPY optimisation with regulatory constraints (Basel III RWA, LCR) and an internal DV01 limit, showing which constraints bind and why.

### Day 2 — `fixed_income_treasury_analytics.ipynb`
Carry decomposition across a three-bond UK gilt portfolio, a deterministic OCI scenario tool (IFRS 9 / CET1 impact of a +100bps parallel shift), and a Monte Carlo framework (1,000 simulations) producing a full CET1 distribution with tail statistics. An interactive Streamlit version of the OCI tool is in `../Apps/oci_tool.py`.

### Day 3 — `fixed_income_convexity_alm.ipynb`
Convexity as a second-order correction to DV01-based P&L, the NII vs OCI trade-off as the binding ALM constraint in a banking book, and a Hidden Markov Model applied to interest rate regime detection — framed as a direct translation of the FX vol regime framework to central bank cycle identification.

---

## Dependencies

```
numpy  pandas  matplotlib  cvxpy  hmmlearn  scipy  streamlit
```

Custom bond analytics (`compute_dv01`, `compute_convexity`) live in `../FixedIncome/bonds.py`.

## DV01 Conventions

Day 1 computes portfolio-level DV01 in £m per basis point using `notional × modified_duration × 0.0001`. Days 2–3 use bump-and-reprice via `compute_dv01`, returning £ per £100 face per basis point. Both are standard approaches; the difference reflects portfolio-level vs. instrument-level analysis.
