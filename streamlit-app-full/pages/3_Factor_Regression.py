# pages/3_Factor_Regression.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional statsmodels for t-stats & HAC. Fallback to NumPy OLS.
try:
    import statsmodels.api as sm  # type: ignore
except Exception:
    sm = None

st.set_page_config(page_title="Factor Regression", layout="wide")
st.title("Factor Regression")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

FACTOR_COLUMNS = [
    "S&P500", "Credit", "Value", "Growth", "Momentum", "Size", "Quality", "Carry"
]

def read_factors_excel_prices(file_or_path) -> pd.DataFrame:
    """
    Read factor PRICES from Excel with this layout:
      - Data starts on row 7 (1-based) -> skiprows=6
      - Column A: Date
      - Columns B..I: S&P500, Credit, Value, Growth, Momentum, Size, Quality, Carry
    Returns MONTHLY RETURNS DataFrame indexed by month-end (ME).
    """
    df = pd.read_excel(file_or_path, header=None, skiprows=6)
    df = df.iloc[:, : 1 + len(FACTOR_COLUMNS)].copy()

    cols = ["Date"] + FACTOR_COLUMNS
    df.columns = cols[: df.shape[1]]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    for c in [c for c in df.columns if c in FACTOR_COLUMNS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Prices -> month-end prices -> monthly returns
    px_last = df.resample("ME").last()
    ret = px_last.pct_change().dropna(how="all")

    # keep only expected factor columns, drop all-NaN columns
    ret = ret[[c for c in FACTOR_COLUMNS if c in ret.columns]]
    ret = ret.dropna(how="all")
    return ret

def align_target_and_factors(r: pd.Series, F: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    r = r.dropna()
    F = F.dropna(how="all")
    idx = r.index.intersection(F.index)
    y = r.reindex(idx)
    X = F.reindex(idx).dropna(how="all")
    y = y.reindex(X.index)
    return y, X

def run_ols(y: pd.Series, X: pd.DataFrame, add_const: bool = True, nw_lags: Optional[int] = None) -> Dict:
    if sm is None:
        Xv = X.values
        if add_const:
            Xv = np.c_[np.ones(len(Xv)), Xv]
        beta = np.linalg.lstsq(Xv, y.values, rcond=None)[0]
        resid = y.values - Xv.dot(beta)
        r2 = 1.0 - (np.var(resid, ddof=Xv.shape[1]) / np.var(y.values, ddof=1)) if y.var(ddof=1) != 0 else np.nan
        if add_const:
            intercept = float(beta[0]); b = beta[1:]
        else:
            intercept = np.nan; b = beta
        return {
            "betas": pd.Series(b, index=X.columns, name="beta"),
            "tstats": pd.Series(index=X.columns, dtype=float),
            "intercept": intercept,
            "intercept_t": np.nan,
            "r2": float(r2),
            "resid": pd.Series(resid, index=y.index, name="resid"),
        }

    X1 = sm.add_constant(X) if add_const else X
    if nw_lags is not None and nw_lags > 0:
        model = sm.OLS(y, X1).fit(cov_type="HAC", cov_kwds={"maxlags": int(nw_lags)})
    else:
        model = sm.OLS(y, X1).fit()
    params, tvals = model.params, model.tvalues
    intercept = params.get("const", np.nan)
    intercept_t = tvals.get("const", np.nan)
    betas = params.drop("const", errors="ignore").rename("beta")
    tstats = tvals.drop("const", errors="ignore").rename("t")
    return {
        "betas": betas,
        "tstats": tstats,
        "intercept": float(intercept),
        "intercept_t": float(intercept_t),
        "r2": float(model.rsquared),
        "resid": model.resid.rename("resid"),
    }

def rolling_betas(r: pd.Series, F: pd.DataFrame, window: int = 36, min_obs: int = 24,
                  add_const: bool = True, nw_lags: Optional[int] = None) -> pd.DataFrame:
    rows, idx = [], []
    y_full, X_full = align_target_and_factors(r, F)
    for end in range(window, len(y_full) + 1):
        y = y_full.iloc[end - window : end]
        X = X_full.loc[y.index]
        if len(y) < min_obs:
            continue
        res = run_ols(y, X, add_const=add_const, nw_lags=nw_lags)
        rows.append(res["betas"])
        idx.append(y.index[-1])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=idx).sort_index()

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.sidebar.write("**Target returns**: use the portfolio you computed in Portfolio Explorer, select a manager from session, or upload a returns file.")

opt = st.radio(
    "Target returns source:",
    ["Portfolio from session", "Manager from session", "Upload returns (CSV/XLSX)"],
    horizontal=True,
)

ret = None
if opt == "Portfolio from session":
    ret = st.session_state.get("_port")
    if isinstance(ret, pd.Series) and not ret.empty:
        st.success(f"Using portfolio from session: {ret.index.min().date()} → {ret.index.max().date()}")
    else:
        st.warning("No portfolio in session. Run Portfolio Explorer first, or choose another source.")
elif opt == "Manager from session":
    panel = st.session_state.get("_panel")
    if isinstance(panel, pd.DataFrame) and not panel.empty:
        col = st.selectbox("Manager", list(panel.columns))
        s = pd.to_numeric(panel[col], errors="coerce").dropna()
        ret = s
        st.success(f"Using manager '{col}': {ret.index.min().date()} → {ret.index.max().date()}")
    else:
        st.warning("No panel in session. Run Portfolio Explorer first, or upload returns.")
else:
    upl = st.file_uploader("Upload returns CSV/XLSX (Date + one return column)", type=["csv", "xlsx", "xls"])
    if upl:
        try:
            df_r = pd.read_csv(upl)
        except Exception:
            upl.seek(0)
            df_r = pd.read_excel(upl)
        df_r = df_r.dropna(how="all").dropna(axis=1, how="all")
        cand = [c for c in df_r.columns if "date" in str(c).lower()] or [df_r.columns[0]]
        date_col = cand[0]
        df_r[date_col] = pd.to_datetime(df_r[date_col], errors="coerce")
        df_r = df_r.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        num_cols = [c for c in df_r.columns if pd.api.types.is_numeric_dtype(df_r[c])]
        if not num_cols:
            st.error("Could not find a numeric returns column.")
        else:
            s = pd.to_numeric(df_r[num_cols[0]], errors="coerce").dropna()
            if s.abs().max() > 2.0:
                s = s / 100.0
            s = (1.0 + s).resample("ME").prod() - 1.0
            ret = s
            st.success(f"Returns loaded: {ret.index.min().date()} → {ret.index.max().date()}")

# --- Factor auto-load with optional override ---
st.subheader("Factors (auto‑loaded from data/Factor Returns.xlsx)")

default_path = Path("data") / "Factor Returns.xlsx"
factors = None
if default_path.exists():
    try:
        factors = read_factors_excel_prices(default_path)
        st.success(
            f"Loaded factors from data/Factor Returns.xlsx: "
            f"{factors.index.min().date()} → {factors.index.max().date()} "
            f"({len(factors)} months; cols={list(factors.columns)})"
        )
        st.dataframe(factors.tail().style.format("{:.4f}"), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to parse default factor Excel: {e}")

# Optional override
st.caption("Optionally upload a factor **prices** Excel to override the default.")
upl_factors = st.file_uploader(
    "Override factor file (optional): data start Row 7; Date in Col A; B..I: S&P500, Credit, Value, Growth, Momentum, Size, Quality, Carry",
    type=["xlsx", "xls"],
)
if upl_factors is not None:
    try:
        factors = read_factors_excel_prices(upl_factors)
        st.info("Overrode default factors with uploaded file.")
        st.dataframe(factors.tail().style.format("{:.4f}"), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to parse uploaded factor Excel: {e}")

c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
with c1:
    add_const = st.checkbox("Include intercept", True)
with c2:
    use_hac = st.checkbox("HAC (Newey–West)", True)
with c3:
    nw_lags = st.number_input("NW lags", 0, 24, 6, 1)
with c4:
    roll = st.number_input("Rolling window (months)", 12, 120, 36, 6)

run = st.button("Run regression", type="primary", disabled=(ret is None or factors is None))

if run and ret is not None and factors is not None:
    y, X = align_target_and_factors(ret, factors)
    if X.empty or y.empty:
        st.warning("No overlapping dates between target returns and factor returns.")
    else:
        res = run_ols(y, X, add_const=add_const, nw_lags=(int(nw_lags) if use_hac else None))

        betas = res["betas"].to_frame("beta")
        tstats = res["tstats"].to_frame("t")
        summary = betas.join(tstats, how="outer")

        l, r = st.columns(2)
        with l:
            st.metric("R²", f"{res['r2']:.3f}")
            if add_const:
                st.metric("Intercept", f"{res['intercept']:.4f}  (t={res['intercept_t']:.2f})")
            st.dataframe(summary.style.format({"beta": "{:.4f}", "t": "{:.2f}"}), use_container_width=True)
        with r:
            if not summary.empty:
                fig = px.bar(
                    summary.reset_index(),
                    x="index",
                    y="beta",
                    color="t",
                    title="Factor Betas (color=t)",
                    color_continuous_scale="RdBu",
                )
                fig.update_layout(xaxis_title="", yaxis_title="Beta")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rolling betas")
        betas_ts = rolling_betas(
            ret, factors, window=int(roll), min_obs=max(int(roll * 2 / 3), 12), add_const=add_const,
            nw_lags=(int(nw_lags) if use_hac else None)
        )
        if not betas_ts.empty:
            st.dataframe(betas_ts.tail().style.format("{:.3f}"), use_container_width=True)
            fig2 = go.Figure()
            for c in betas_ts.columns:
                fig2.add_trace(go.Scatter(x=betas_ts.index, y=betas_ts[c], mode="lines", name=c))
            fig2.update_layout(title="Rolling Betas", yaxis_title="Beta", xaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Residuals")
        resid = res["resid"]
        st.dataframe(resid.to_frame().tail().style.format("{:.4f}"), use_container_width=True)
        st.plotly_chart(px.histogram(resid, nbins=30, title="Residuals Distribution"), use_container_width=True)
