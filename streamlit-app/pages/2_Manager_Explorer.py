
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.ingest import read_manager_sheet
from src.metrics import summarize, compute_drawdown
from src.hedging import apply_partial_hedge

st.set_page_config(page_title="Manager Explorer", layout="wide")
st.title("Manager Explorer")

up = st.file_uploader("Upload manager returns (CSV/XLSX)", type=["csv","xlsx","xls"])

df = read_manager_sheet(up) if up else read_manager_sheet(None)
if df.empty:
    st.info("Upload a file or use the demo dataset.")
    st.stop()

m = st.selectbox("Manager", df.columns)
hedge = st.slider("Partial hedge (0-100%)", 0, 100, 0, 5) / 100.0

series = df[m]
adj = apply_partial_hedge(series, hedge_weight=hedge)

# Plots and tables
c1, c2 = st.columns(2)
with c1:
    st.subheader("Summary")
    st.dataframe(summarize(adj).style.format("{:.2%}"), use_container_width=True)
with c2:
    st.subheader("Drawdown")
    dd = compute_drawdown(adj)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, mode='lines', name='DD'))
    fig.update_layout(yaxis_title='Drawdown', xaxis_title='', yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
