
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.ingest import read_manager_sheet
from src.metrics import summarize

st.set_page_config(page_title="Portfolio Explorer", layout="wide")
st.title("Portfolio Explorer")

st.sidebar.header("Inputs")
up = st.sidebar.file_uploader("Upload manager returns (CSV/XLSX)", type=["csv","xlsx","xls"])

df = read_manager_sheet(up) if up else read_manager_sheet(None)

if df.empty:
    st.info("No data available. Upload a file or use the demo dataset.")
    st.stop()

managers = list(df.columns)
sel = st.multiselect("Select managers", managers, default=managers[: min(5, len(managers))])

if not sel:
    st.warning("Select at least one manager.")
    st.stop()

w_cols = st.columns(len(sel))
weights = []
for i, m in enumerate(sel):
    with w_cols[i]:
        weights.append(st.number_input(f"{m} wgt", value=1.0/len(sel), min_value=0.0, max_value=1.0, step=0.01))

w = np.array(weights)
if w.sum() == 0:
    st.error("Weights sum to 0.")
    st.stop()

w = w / w.sum()
port_ret = (df[sel] @ w).rename("Portfolio")

# Save to session for Factor page
st.session_state["portfolio_returns"] = port_ret

# Show monthly heatmap (Year x Month) with Pandas Styler
piv = port_ret.to_frame()
piv['Year'] = piv.index.year
piv['Month'] = piv.index.month
piv = piv.pivot_table(index='Year', columns='Month', values='Portfolio', aggfunc='mean')

st.subheader("Monthly Heatmap")
st.dataframe(piv.style.format("{:.1%}").background_gradient(axis=None, cmap="RdYlGn"), use_container_width=True)

# Performance summary
st.subheader("Summary")
sum_df = summarize(port_ret)
st.dataframe(sum_df.style.format("{:.2%}"), use_container_width=True)

# Equity curve
st.subheader("Equity curve")
curve = (1 + port_ret.fillna(0)).cumprod()
fig = px.line(curve, title="Cumulative Growth of 1")
fig.update_layout(yaxis_title="Multiple", xaxis_title="")
st.plotly_chart(fig, use_container_width=True)
