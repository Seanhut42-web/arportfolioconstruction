# pages/4_Optimizer.py
from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.metrics import summarize, compute_drawdown
from src.ingest import parse_workbook

# Optional SciPy for SLSQP; fallback if missing
try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

st.set_page_config(page_title="Optimizer", layout="wide")
st.title("Portfolio Optimizer")

st.markdown(
    """
    **Workflow**: Run *Portfolio Explorer* first to build a panel for your chosen managers and hedging setup. 
    This page will pick that up from session. If not found, it will parse the workbook at `data/Manager Track Records v2.xlsx`
    as a fallback (GBP base).
    """
)

# --- Load panel either from session or workbook fallback
panel = st.session_state.get("_panel")
if not isinstance(panel, pd.DataFrame) or panel.empty:
    try:
        panel = parse_workbook(Path("data") / "Manager Track Records v2.xlsx")
        st.info("Loaded panel from data/Manager Track Records v2.xlsx (fallback).")
    except Exception:
        st.error("No panel in session and failed to load workbook. Open Portfolio Explorer first.")
        st.stop()

panel = panel.dropna(how="all")
managers = list(panel.columns)

st.subheader("Selection & Bounds")

# Build editable table (initial view). Users can tweak and then press Optimize.
df_setup = pd.DataFrame(
    {
        "Include": [True] * len(managers),
        "Manager": managers,
        "Min": [0.0] * len(managers),
        "Max": [1.0] * len(managers),
        "Lock": [False] * len(managers),
        "Weight": [round(1.0 / len(managers), 4)] * len(managers),
    }
)

edited = st.data_editor(
    df_setup,
    hide_index=True,
    num_rows="fixed",
    column_config={
        "Include": st.column_config.CheckboxColumn(help="Include manager in optimization"),
        "Manager": st.column_config.TextColumn(disabled=True),
        "Min": st.column_config.NumberColumn(format="%.3f"),
        "Max": st.column_config.NumberColumn(format="%.3f"),
        "Lock": st.column_config.CheckboxColumn(help="Fix weight to the value in the Weight column"),
        "Weight": st.column_config.NumberColumn(
            format="%.4f", help="Only used if Lock=True; also used as initial guess"
        ),
    },
    use_container_width=True,
)

obj = st.selectbox(
    "Objective",
    ["Max Return", "Max Sharpe", "Max Calmar", "Min Drawdown", "Min Volatility"],
    index=1,
)

btn = st.button("Optimize", type="primary")

# --- Helper functions used by the objective
def annualize_std(rm: pd.Series) -> float:
    return float(rm.std(ddof=0) * math.sqrt(12))


def cagr_from_series(rm: pd.Series) -> float:
    if rm.empty:
        return float("nan")
    cg = (1.0 + rm).prod()
    yrs = len(rm) / 12.0
    return float(cg ** (1 / yrs) - 1) if yrs > 0 else float("nan")


def max_drawdown_from_series(rm: pd.Series) -> float:
    cum = (1.0 + rm).cumprod()
    return float(compute_drawdown(cum).min())

# --- Run optimizer on click
if btn:
    df = edited.copy()
    df = df[df["Include"]]
    if df.empty:
        st.warning("Select at least one manager.")
        st.stop()

    cols = df["Manager"].tolist()
    R = panel[cols].dropna(how="all").fillna(0.0)

    # Initial guess (normalized from Weight column over included)
    w0 = df["Weight"].values.astype(float)
    w0 = np.ones(len(w0)) / len(w0) if w0.sum() == 0 else (w0 / w0.sum())

    # Bounds and locks
    lb = df["Min"].values.astype(float)
    ub = df["Max"].values.astype(float)
    lock = df["Lock"].values.astype(bool)
    w_lock = df["Weight"].values.astype(float)
    for i, L in enumerate(lock):
        if L:
            lb[i] = w_lock[i]
            ub[i] = w_lock[i]
            w0[i] = w_lock[i]

    # Equality: sum weights = 1
    def cons_sum(w):
        return np.sum(w) - 1.0

    def obj_value(w):
        rp = (R @ w).rename("p")
        if obj == "Max Return":
            return -rp.mean()  # maximize mean monthly
        elif obj == "Max Sharpe":
            vol = rp.std(ddof=0)
            if vol == 0:
                return 1e3
            return -(rp.mean() * math.sqrt(12) / (vol * math.sqrt(12)))
        elif obj == "Max Calmar":
            cagr = cagr_from_series(rp)
            mdd = abs(max_drawdown_from_series(rp))
            if not np.isfinite(cagr) or mdd == 0:
                return 1e3
            return -(cagr / mdd)
        elif obj == "Min Drawdown":
            return abs(max_drawdown_from_series(rp))
        elif obj == "Min Volatility":
            return annualize_std(rp)
        return 0.0

    if minimize is None:
        st.warning(
            "SciPy not available. Using a heuristic search (coarse grid) — "
            "consider adding scipy to requirements for best results."
        )
        # Simple heuristic: project w0 into bounds, normalize; local random search
        w = np.clip(w0, lb, ub)
        w = (np.ones_like(w) / len(w)) if w.sum() == 0 else (w / w.sum())
        best_w, best_v = w.copy(), obj_value(w)
        rng = np.random.default_rng(42)
        for _ in range(2000):
            step = rng.normal(scale=0.02, size=len(w))
            w_try = np.clip(w + step, lb, ub)
            if w_try.sum() == 0:
                continue
            w_try = w_try / w_try.sum()
            v = obj_value(w_try)
            if v < best_v:
                best_v, best_w = v, w_try
                w = w_try
        w_opt = best_w
    else:
        cons = ({"type": "eq", "fun": cons_sum},)
        bounds = list(zip(lb, ub))
        res = minimize(
            lambda w: obj_value(w),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000},
        )
        if not res.success:
            st.warning(f"Optimizer status: {res.message}")
        w_opt = np.clip(res.x, lb, ub)
        if w_opt.sum() != 0:
            w_opt = w_opt / w_opt.sum()

    # Persist the solution so Analysis survives UI reruns (radio changes)
    st.session_state["_opt_cols"] = cols
    st.session_state["_opt_weights"] = w_opt
    st.session_state["_opt_panel"] = panel[cols]
    st.session_state["_opt_port"] = (panel[cols] @ w_opt).dropna().rename("Portfolio (Optimized)")

# --- Always-render Analysis, using persisted optimized results if present
st.subheader("Analysis")

opt_cols = st.session_state.get("_opt_cols")
opt_weights = st.session_state.get("_opt_weights")
opt_panel = st.session_state.get("_opt_panel")
opt_port = st.session_state.get("_opt_port")

if opt_port is None or opt_panel is None or opt_cols is None or opt_weights is None:
    st.info("Run **Optimize** to see results.")
else:
    # Output weights
    out = pd.DataFrame({"Manager": opt_cols, "Weight": opt_weights})
    st.markdown("**Optimal Weights**")
    st.dataframe(out.style.format({"Weight": "{:.2%}"}), use_container_width=True)

    chart = st.radio(
        "Chart",
        ["Summary", "Cumulative", "Drawdown", "12M Return", "12M Vol", "Monthly Bars", "Correlation", "Year×Month"],
        index=0,
        horizontal=True,
    )

    port = opt_port

    if chart == "Summary":
        stats = summarize(port)
        df_stats = pd.DataFrame(stats, index=[0]).T.rename(columns={0: "Value"})
        st.dataframe(df_stats.style.format({"Value": "{:.2%}"}), use_container_width=True)
        st.markdown("**Active Weights**")
        st.dataframe(out.style.format({"Weight": "{:.2%}"}), use_container_width=True)

    elif chart == "Cumulative":
        fig = go.Figure()
        cum_port = (1.0 + port).cumprod()
        fig.add_trace(
            go.Scatter(x=cum_port.index, y=cum_port.values, name="Optimized", line=dict(width=3, color="black"))
        )
        for m in opt_cols:
            s = opt_panel[m].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=(1 + s).cumprod(), name=m, line=dict(width=1), opacity=0.4))
        fig.update_layout(
            title="Cumulative Growth of £1",
            hovermode="x unified",
            legend=dict(orientation="h"),
            yaxis_title="Value (£)",
            xaxis_title="Date",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Drawdown":
        cum_port = (1.0 + port).cumprod()
        dd = compute_drawdown(cum_port)
        y_min = min(-1.0, float(dd.min()) * 1.05) if np.isfinite(dd.min()) else -1.0
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", line=dict(color="#d62728", width=2)))
        fig.update_layout(
            title="Portfolio Drawdown",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        fig.update_yaxes(tickformat=".0%", range=[y_min, 0])
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "12M Return":
        roll12_ret = (1.0 + port).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0
        if roll12_ret.dropna().empty:
            st.info("Not enough data (need ≥12 months).")
        else:
            fig = go.Figure(
                go.Scatter(
                    x=roll12_ret.index, y=roll12_ret.values, name="12M Rolling Return", line=dict(color="#1f77b4", width=2)
                )
            )
            fig.update_layout(
                title="12‑month Rolling Return",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Return (12M)",
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "12M Vol":
        roll12_vol = port.rolling(12, min_periods=12).std(ddof=0) * math.sqrt(12)
        if roll12_vol.dropna().empty:
            st.info("Not enough data (need ≥12 months).")
        else:
            fig = go.Figure(
                go.Scatter(
                    x=roll12_vol.index, y=roll12_vol.values, name="12M Rolling Vol (ann.)", line=dict(color="#ff7f0e", width=2)
                )
            )
            fig.update_layout(
                title="12‑month Rolling Volatility (Annualised)",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Volatility (ann.)",
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Monthly Bars":
        dfb = port.to_frame("Monthly Return").reset_index(names="Month")
        fig_bar = px.bar(
            dfb,
            x="Month",
            y="Monthly Return",
            title="Portfolio Monthly Returns",
            color="Monthly Return",
            color_continuous_scale="RdYlGn",
        )
        fig_bar.update_yaxes(tickformat=".0%")
        fig_bar.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)

    elif chart == "Correlation":
        filt = opt_panel.copy()
        valid_cols = [c for c in filt.columns if filt[c].notna().any()]
        filt = filt[valid_cols]
        if len(valid_cols) >= 2:
            corr = filt.corr(min_periods=3).dropna(how="all").dropna(how="all", axis=1)
            size = max(420, 70 * corr.shape[0])
            fig_corr = px.imshow(
                corr.round(2),
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation (monthly returns)",
            )
            fig_corr.update_layout(width=size, height=size, margin=dict(l=60, r=20, t=60, b=60))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select ≥2 managers with overlapping data to display correlation.")

    elif chart == "Year×Month":
        dfym = port.to_frame("ret")
        dfym["Year"] = dfym.index.year
        dfym["Month"] = dfym.index.strftime("%b")
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        piv = dfym.pivot(index="Year", columns="Month", values="ret")
        piv = piv[[c for c in order if c in piv.columns]]
        if piv.empty:
            st.info("No data to build Year × Month table.")
        else:
            fig_hm = px.imshow(piv, color_continuous_scale="RdYlGn", origin="upper")
            fig_hm.update_traces(
                hovertemplate="Year=%{y}<br>Month=%{x}<br>Return=%{z:.1%}<extra></extra>"
            )
            fig_hm.update_traces(text=piv.applymap(lambda v: f"{v:.1%}").values, texttemplate="%{text}")
            fig_hm.update_layout(
                title="Year × Month (Portfolio monthly returns)",
                xaxis_title="Month",
                yaxis_title="Year",
                coloraxis_colorbar=dict(title="Return", tickformat=".0%"),
                margin=dict(l=60, r=20, t=60, b=60),
            )
            st.plotly_chart(fig_hm, use_container_width=True)
