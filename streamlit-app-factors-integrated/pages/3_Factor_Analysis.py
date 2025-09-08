from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.hedging import build_hedging_inputs
from src.factors import (
    load_integrated_factors, newey_west_ols, compute_vif, rolling_betas,
    series_gbp_for_manager
)

st.set_page_config(page_title="Factor Analysis", layout="wide")

@st.cache_data(show_spinner="Loading manager data…", ttl=None)
def load_managers():
    xlsx_path = Path(__file__).parent.parent / "data" / "Manager Track Records v2.xlsx"
    return build_hedging_inputs(xlsx_path)

man_local_m, man_ccy, fx_ret_m, span = load_managers()

# Load integrated factors
@st.cache_data(show_spinner="Loading integrated factors…", ttl=None)
def load_factors():
    return load_integrated_factors(Path(__file__).parent.parent / 'data')

fac_m = load_factors()
st.header("Factor Analysis")
st.success(f"Loaded {fac_m.shape[1]} factors | {fac_m.index.min().date()} → {fac_m.index.max().date()}")

st.subheader("Diagnostics")
if fac_m.shape[1] >= 2:
    corr = fac_m.corr()
    figc = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='Factor Correlation')
    st.plotly_chart(figc, use_container_width=True)
    try:
        vifs = compute_vif(fac_m)
        st.caption("Variance Inflation Factors (higher → more collinearity)")
        st.dataframe(vifs.to_frame().style.format("{:.2f}"))
    except Exception:
        pass

# ---- Manager exposures ----
st.subheader("Manager exposures (current hedging settings)")
mgr = st.selectbox("Manager", sorted(man_local_m.keys()))
start_ts = st.session_state.get('_start_ts', fac_m.index.min())
params = st.session_state.get('_params', dict(fx_mode='Fully hedged', hedge_ratio=1.0, gbp_cash_ann=0.05, usd_cash_ann=0.05))

s_mgr = series_gbp_for_manager(mgr, pd.Timestamp(start_ts), params, man_local_m, man_ccy, fx_ret_m)
y = s_mgr.reindex(fac_m.index)
X = fac_m.loc[y.index]
model = newey_west_ols(y, X, add_const=True, maxlags=3)
if model is None:
    st.warning("Not enough overlapping data for regression.")
else:
    params_df = model.params.rename('beta').to_frame()
    if 'const' in params_df.index:
        alpha = params_df.loc['const', 'beta']
        params_df = params_df.drop(index='const')
    else:
        alpha = np.nan
    tvals = model.tvalues.drop('const', errors='ignore').rename('t')
    res_df = params_df.join(tvals, how='left')
    res_df['abs(t)'] = res_df['t'].abs()
    st.markdown(f"**Alpha (monthly):** {alpha:.3%} | **R²:** {model.rsquared:.2f}")
    st.dataframe(res_df.sort_values('abs(t)', ascending=False).drop(columns=['abs(t)']).style.format({'beta':'{:.2f}','t':'{:.2f}'}))

    fig = go.Figure(go.Bar(x=res_df.index, y=res_df['beta']))
    fig.update_layout(title=f"{mgr} betas (OLS, Newey–West)", yaxis_title='Beta')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Rolling betas (36M)**")
    roll = rolling_betas(y, X, window=36, add_const=True)
    if roll.empty:
        st.info("Not enough data for 36M rolling window.")
    else:
        fig_r = go.Figure()
        for c in roll.columns:
            fig_r.add_trace(go.Scatter(x=roll.index, y=roll[c], name=c))
        fig_r.update_layout(hovermode='x unified', yaxis_title='Beta', title=f"Rolling betas: {mgr}")
        st.plotly_chart(fig_r, use_container_width=True)

# ---- Portfolio exposures ----
st.subheader("Current portfolio exposures (from Portfolio Explorer run)")
port = st.session_state.get('_port')
if port is None or port.empty:
    st.info("Run the Portfolio Explorer first to create a portfolio series.")
else:
    y = port.reindex(fac_m.index)
    X = fac_m.loc[y.index]
    model = newey_west_ols(y, X, add_const=True, maxlags=3)
    if model is None:
        st.warning("Not enough overlapping data for regression.")
    else:
        params_df = model.params.rename('beta').to_frame()
        alpha = params_df.pop('const').values[0] if 'const' in params_df.index else np.nan
        tvals = model.tvalues.drop('const', errors='ignore').rename('t')
        res_df = params_df.join(tvals, how='left')
        res_df['abs(t)'] = res_df['t'].abs()
        st.markdown(f"**Alpha (monthly):** {alpha:.3%} | **R²:** {model.rsquared:.2f}")
        st.dataframe(res_df.sort_values('abs(t)', ascending=False).drop(columns=['abs(t)']).style.format({'beta':'{:.2f}','t':'{:.2f}'}))

        fig = go.Figure(go.Bar(x=res_df.index, y=res_df['beta']))
        fig.update_layout(title=f"Portfolio betas (OLS, Newey–West)", yaxis_title='Beta')
        st.plotly_chart(fig, use_container_width=True)
