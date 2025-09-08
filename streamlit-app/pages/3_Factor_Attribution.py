
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.factors import read_factor_data, run_factor_regression, rolling_factor_regression

st.set_page_config(page_title="Factor Attribution", layout="wide")
st.title("Factor Attribution")

with st.expander("How this works", expanded=False):
    st.markdown("""
    Upload a factor dataset (CSV or Excel) with a **Date** column and factor columns in **decimal returns**.
    Select the portfolio or a single return series, and run OLS (optionally with Newey–West).
    """)

# Target returns
target_source = st.radio("Target returns series:", ["Portfolio (session)", "Upload returns CSV/XLSX"], horizontal=True)
ret_series = None

if target_source == "Portfolio (session)":
    r = st.session_state.get("portfolio_returns", None)
    if isinstance(r, pd.Series) and not r.empty:
        ret_series = r
        st.success(f"Using portfolio returns from session: {ret_series.index.min().date()} → {ret_series.index.max().date()}")
    else:
        st.warning("No portfolio returns in session. Go to **Portfolio Explorer** first.")
else:
    up_ret = st.file_uploader("Upload returns series (CSV or Excel)", type=["csv","xlsx","xls"], key="ret_upl")
    if up_ret:
        try:
            r = pd.read_csv(up_ret)
        except Exception:
            up_ret.seek(0)
            r = pd.read_excel(up_ret)
        r.columns = [str(c).strip() for c in r.columns]
        date_col = "Date" if "Date" in r.columns else r.columns[0]
        r[date_col] = pd.to_datetime(r[date_col], errors="coerce")
        r = r.dropna(subset=[date_col]).set_index(date_col).sort_index()
        value_col = [c for c in r.columns if c.lower() in ("ret","return","returns")]
        if not value_col:
            value_col = [c for c in r.columns if c != date_col][:1]
        ret_series = pd.to_numeric(r[value_col[0]], errors="coerce").dropna()
        if ret_series.abs().max() > 2.0:
            ret_series = ret_series / 100.0
        st.success(f"Returns loaded: {ret_series.index.min().date()} → {ret_series.index.max().date()}")

# Factor data
st.subheader("Factors")
up_factors = st.file_uploader("Upload factor set (CSV or Excel)", type=["csv","xlsx","xls"]) 
# Provide default path
if up_factors is None:
    try:
        default_path = "streamlit-app/data/factors/Factor_Returns_standardized.csv"
        factors_df = pd.read_csv(default_path, parse_dates=[0])
        factors_df = factors_df.set_index(factors_df.columns[0])
    except Exception:
        factors_df = None
else:
    factors_df = read_factor_data(up_factors)

if factors_df is not None:
    st.dataframe(factors_df.tail().style.format("{:.4f}"), use_container_width=True)

colA, colB, colC, colD = st.columns([1,1,1,1.2])
with colA:
    add_const = st.checkbox("Include intercept", value=True)
with colB:
    use_hac = st.checkbox("HAC (Newey–West)", value=True)
with colC:
    nw_lags = st.number_input("NW lags", min_value=0, max_value=24, value=6, step=1)
with colD:
    roll_win = st.number_input("Rolling window (months)", min_value=12, max_value=120, value=36, step=6)

run = st.button("Run factor regression", type="primary", disabled=ret_series is None or factors_df is None)

if run and ret_series is not None and factors_df is not None:
    res = run_factor_regression(ret_series, factors_df, add_const=add_const, nw_lags=(nw_lags if use_hac else None))
    betas = res["betas"].to_frame("beta"); tstats = res["tstats"].to_frame("t")
    summary = betas.join(tstats, how="outer")

    m1, m2 = st.columns([1,1])
    with m1:
        st.metric("R²", f"{res['r2']:.3f}")
        if add_const:
            st.metric("Intercept", f"{res['intercept']:.4f}  (t={res['intercept_t']:.2f})")
        st.dataframe(summary.style.format({"beta":"{:.4f}","t":"{:.2f}"}), use_container_width=True)
    with m2:
        fig = px.bar(summary.reset_index(), x="index", y="beta", color="t", title="Factor Betas (color: t)", color_continuous_scale="RdBu")
        fig.update_layout(xaxis_title="", yaxis_title="Beta")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling betas")
    betas_ts = rolling_factor_regression(ret_series, factors_df, window=int(roll_win), min_obs=max(int(roll_win*2/3),12), add_const=add_const, nw_lags=(nw_lags if use_hac else None))
    if not betas_ts.empty:
        st.dataframe(betas_ts.tail().style.format("{:.3f}"), use_container_width=True)
        fig2 = go.Figure()
        for col in betas_ts.columns:
            fig2.add_trace(go.Scatter(x=betas_ts.index, y=betas_ts[col], mode="lines", name=col))
        fig2.update_layout(title="Rolling Betas", yaxis_title="Beta", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Residuals")
    resid = res["residuals"]
    st.dataframe(resid.to_frame().tail().style.format("{:.4f}"), use_container_width=True)
    st.plotly_chart(px.histogram(resid, nbins=30, title="Residuals Distribution"), use_container_width=True)
