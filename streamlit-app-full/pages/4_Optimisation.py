from __future__ import annotations
import streamlit as st
import pandas as pd

from src.alloc import estimate_mu_cov, optimise_mvo, solve_risk_parity
from src.state import apply_theme, get_plotly_template

st.set_page_config(page_title="Optimisation", layout="wide")
st.title("Optimisation")

with st.sidebar:
    dark = st.checkbox("Dark mode", value=st.session_state.get("_dark_mode", False))
apply_theme(dark)
template = get_plotly_template()

panel = st.session_state.get("_panel")
chosen = st.session_state.get("_chosen", [])
if panel is None or not chosen:
    st.info("Build a panel in Portfolio Explorer first.")
    st.stop()

sub = panel[chosen].dropna(how="all")
mu, Sigma = estimate_mu_cov(sub)

st.subheader("Mean–Variance Optimisation")
c1, c2, c3 = st.columns(3)
with c1:
    objective = st.selectbox("Objective", ["max_sharpe", "target_return", "target_vol"])
with c2:
    target_ret = st.number_input("Target return (ann.)", value=0.06, step=0.005, format="%.3f")
with c3:
    target_vol = st.number_input("Target vol (ann.)", value=0.10, step=0.005, format="%.3f")
bounds = st.slider("Bounds", 0.0, 1.0, (0.0, 1.0), 0.05)

if st.button("Solve MVO", type="primary"):
    w_mvo = optimise_mvo(mu, Sigma, objective=objective,
                         target_ret=target_ret, target_vol=target_vol, bounds=bounds)
    st.dataframe(w_mvo.to_frame("Weight").style.format("{:.2%}"))
    if st.button("Apply to Portfolio Explorer"):
        st.session_state["_weights"] = w_mvo
        st.success("Weights sent to Portfolio Explorer.")

st.subheader("Risk Parity (ERC)")
if st.button("Solve Risk Parity"):
    w_rpc = solve_risk_parity(Sigma, bounds=bounds)
    st.dataframe(w_rpc.to_frame("Weight").style.format("{:.2%}"))
    if st.button("Apply to Portfolio Explorer", key="apply_rpc"):
        st.session_state["_weights"] = w_rpc
        st.success("Weights sent to Portfolio Explorer.")
