import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("OCI Scenario Tool")
st.markdown(
    "Interactive companion to **Section 2b** of `fixed_income_treasury_analytics.ipynb`. "
    "Applies a parallel rate shock to a three-bond AFS gilt portfolio and computes the "
    "mark-to-market OCI loss and resulting CET1 impact. Use the sidebar to adjust the "
    "shock size and the illustrative equity/RWA base."
)

# --- Sidebar inputs ---
st.sidebar.header("Parameters")
shock_bps    = st.sidebar.slider("Rate shock (bps)", -200, 200, 100, step=10)
total_equity = st.sidebar.number_input("Equity base (£)", value=100)
rwa          = st.sidebar.number_input("RWA (£)", value=800)

# --- Portfolio (DV01 = £ per £100 face per 1bp, from compute_dv01) ---
portfolio = [
    {"name": "UK Gilt 4% 2029",  "dv01": 0.0433},
    {"name": "UK Gilt 2% 2027",  "dv01": 0.0272},
    {"name": "UK Gilt 5% 2035",  "dv01": 0.0787},
]

# --- Processing ---
names         = [b["name"] for b in portfolio]
dv01s         = np.array([b["dv01"] for b in portfolio])
price_changes = -dv01s * shock_bps
oci_impact    = price_changes.sum()

equity_after = total_equity + oci_impact
cet1_before  = total_equity / rwa * 100
cet1_after   = equity_after / rwa * 100

# --- Metric cards ---
col1, col2, col3 = st.columns(3)
col1.metric("OCI Impact (£)", f"{oci_impact:.2f}")
col2.metric("CET1 Before", f"{cet1_before:.2f}%")
col3.metric("CET1 After", f"{cet1_after:.2f}%",
            delta=f"{(cet1_after - cet1_before):.2f}%")

# --- Per-bond breakdown table ---
df = pd.DataFrame({
    "Bond":          names,
    "DV01 (£/bp)":   [round(d, 4) for d in dv01s],
    "Price change (£)": [round(p, 2) for p in price_changes],
}).set_index("Bond")
st.dataframe(df, use_container_width=True)

# --- Two-panel chart (mirrors notebook Section 2b layout) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: price change per bond
bar_colors = ["#E76F51" if x < 0 else "#4C78A8" for x in price_changes]
axes[0].bar(names, price_changes, color=bar_colors, edgecolor="white")
axes[0].axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
axes[0].set_title(f"Price change per bond ({shock_bps:+d} bps shock)")
axes[0].set_ylabel("£ per £100 face value")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha="right")

# Right: CET1 before vs after with regulatory floor
axes[1].bar(["CET1 before", "CET1 after"],
            [cet1_before, cet1_after],
            color=["#4C78A8", "#E76F51"], edgecolor="white")
axes[1].axhline(10.5, color="#E76F51", linewidth=1.2, linestyle="--",
                label="Regulatory minimum 10.5%")
axes[1].set_title("CET1 impact")
axes[1].set_ylabel("CET1 ratio (%)")
axes[1].legend()

plt.tight_layout()
st.pyplot(fig)

# --- CET1 status banner ---
if cet1_after < 10.5:
    st.error("CET1 breaches regulatory minimum of 10.5% under this scenario")
else:
    st.success("CET1 remains above regulatory minimum of 10.5%")
