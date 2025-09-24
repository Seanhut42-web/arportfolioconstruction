
# === FINAL UPDATED: 2025-09-24 — 0–1 weights, Distribution sub‑tab, Detailed PDF ===
import io
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from src.hedging import build_hedging_inputs, build_panel_for_selection
from src.metrics import summarize, compute_drawdown
from src.contrib import overall_return_contrib, overall_risk_contrib, rolling_contrib
try:
    from src.report import build_pdf as build_pdf_basic
except Exception:
    build_pdf_basic = None
from src.state import load_state_from_query, encode_state_to_query, apply_theme, get_plotly_template

st.set_page_config(page_title="Portfolio Explorer", layout="wide")
load_state_from_query()

@st.cache_data(show_spinner="Preparing hedging inputs…", ttl=None)
def load_inputs():
    xlsx_path = Path(__file__).parent.parent / "data" / "Manager Track Records v2.xlsx"
    return build_hedging_inputs(xlsx_path)

man_local_m, man_ccy, fx_ret_m, span = load_inputs()
manager_options = sorted(list(man_local_m.keys()))

with st.sidebar:
    dark = st.checkbox("Dark mode", value=st.session_state.get("_dark_mode", False))
apply_theme(dark)
template = get_plotly_template()

st.sidebar.header("Selection")
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Select all"):
        for m in manager_options:
            st.session_state[f"mgr_{m}"] = True
with c2:
    if st.button("Clear all"):
        for m in manager_options:
            st.session_state[f"mgr_{m}"] = False

cols = st.sidebar.columns(3)
selected = []
default_selected = set(manager_options[: min(4, len(manager_options))])
for i, m in enumerate(manager_options):
    key = f"mgr_{m}"
    if key not in st.session_state:
        st.session_state[key] = (m in default_selected)
    if cols[i % 3].checkbox(m, value=st.session_state[key], key=key):
        selected.append(m)

normalize = st.sidebar.checkbox("Normalize weights to 100%", value=True)

if selected:
    st.sidebar.subheader("Weights")
weights = {}
if selected:
    w0 = 1.0 / len(selected)
    for m in selected:
        weights[m] = st.sidebar.slider(m, 0.0, 1.0, float(w0), 0.01)

all_months = pd.Index(sorted(set().union(*[s.index for s in man_local_m.values()])))
earliest_date, latest_date = all_months.min().date(), all_months.max().date()
default_start = max(pd.to_datetime("2016-01-01").date(), earliest_date)
start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
    min_value=earliest_date,
    max_value=latest_date,
)

st.sidebar.caption(f"Data span: **{earliest_date}** → **{latest_date}**")

st.sidebar.header("Hedging")
fx_mode = st.sidebar.radio("FX handling", ["Unhedged (spot)", "Fully hedged (CIP proxy)"], index=1)
hedge_ratio = st.sidebar.slider("Hedge ratio (USD exposures)", 0.0, 1.0, 1.0, 0.05)

gbp_cash_ann = st.sidebar.number_input("GBP cash (annualised)", value=0.05, step=0.001, format="%.3f")
usd_cash_ann = st.sidebar.number_input("USD cash (annualised)", value=0.05, step=0.001, format="%.3f")

st.sidebar.caption("Hedged uses monthly carry ≈ (1+GBPcash)/(1+USDcash) − 1. (CIP proxy)")

if st.sidebar.button("Reload data (after changing Excel)"):
    load_inputs.clear()
    st.experimental_rerun()

run = st.button("Run portfolio analytics", type="primary", use_container_width=True)

if run:
    start_ts = pd.Timestamp(start_date)
    panel = build_panel_for_selection(
        man_local_m, man_ccy, fx_ret_m,
        chosen=selected,
        mode="spot" if fx_mode.startswith("Unhedged") else "hedged",
        h_ratio=float(hedge_ratio),
        gbp_ann=float(gbp_cash_ann),
        usd_ann=float(usd_cash_ann),
        start_ts=start_ts,
    )
    w = pd.Series([weights[m] for m in selected], index=selected, dtype=float) if selected else pd.Series(dtype=float)
    if normalize and not w.empty and w.sum() != 0:
        w = w / w.sum()
    port = (panel[selected] * w).sum(axis=1).dropna().rename("Portfolio") if selected else pd.Series(dtype=float)

    st.session_state["_panel"] = panel
    st.session_state["_port"] = port
    st.session_state["_weights"] = w
    st.session_state["_chosen"] = selected
    st.session_state["_start_ts"] = start_ts
    st.session_state["_params"] = dict(
        fx_mode=fx_mode, hedge_ratio=hedge_ratio, gbp_cash_ann=gbp_cash_ann, usd_cash_ann=usd_cash_ann
    )

chart = st.radio(
    "Chart",
    options=[
        "Summary",
        "Cumulative",
        "Drawdown",
        "12M Return",
        "12M Vol",
        "Monthly Bars",
        "Correlation",
        "Year×Month",
    ],
    index=0,
    horizontal=True,
)

panel = st.session_state.get("_panel")
port = st.session_state.get("_port")
chosen = st.session_state.get("_chosen", [])
w = st.session_state.get("_weights")

if port is None or panel is None or not chosen:
    st.info("Select managers, set weights & hedging, pick a start date, then click **Run portfolio analytics**.")
else:
    st.markdown(f"**Period:** {port.index.min().date()} → {port.index.max().date()}")

    if chart == "Summary":
        stats_dict = summarize(port)
        df_stats = pd.DataFrame(stats_dict, index=[0]).T.rename(columns={0: "Value"})
        st.dataframe(
            df_stats.style.format(
                {
                    "Ann. Return": "{:.2%}",
                    "Ann. Vol": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                    "Sharpe (rf=0)": "{:.2f}",
                    "Calmar": "{:.2f}",
                }
            )
        )
        st.markdown("**Active Weights**")
        st.dataframe(pd.DataFrame({"Manager": w.index, "Weight": w.values}).style.format({"Weight": "{:.2%}"}))

    elif chart == "Cumulative":
        fig = go.Figure()
        cum_port = (1.0 + port).cumprod()
        fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, name="Portfolio", line=dict(width=3, color="black")))
        for m in chosen:
            s = panel[m].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=(1 + s).cumprod(), name=m, line=dict(width=1), opacity=0.5))
        fig.update_layout(
            title="Cumulative Growth of £1",
            hovermode="x unified",
            legend=dict(orientation="h"),
            yaxis_title="Value (£)",
            xaxis_title="Date",
            margin=dict(l=40, r=20, t=60, b=40),
            template=template,
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
            template=template,
        )
        fig.update_yaxes(tickformat=".0%", range=[y_min, 0])
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "12M Return":
        roll12_ret = (1.0 + port).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0
        if roll12_ret.dropna().empty:
            st.info("Not enough data (need ≥12 months).")
        else:
            fig = go.Figure(
                go.Scatter(x=roll12_ret.index, y=roll12_ret.values, name="12M Rolling Return", line=dict(color="#1f77b4", width=2))
            )
            fig.update_layout(
                title="12‑month Rolling Return",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Return (12M)",
                margin=dict(l=40, r=20, t=60, b=40),
                template=template,
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "12M Vol":
        roll12_vol = port.rolling(12, min_periods=12).std(ddof=0) * math.sqrt(12)
        if roll12_vol.dropna().empty:
            st.info("Not enough data (need ≥12 months).")
        else:
            fig = go.Figure(
                go.Scatter(x=roll12_vol.index, y=roll12_vol.values, name="12M Rolling Vol (ann.)", line=dict(color="#ff7f0e", width=2))
            )
            fig.update_layout(
                title="12‑month Rolling Volatility (Annualised)",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Volatility (ann.)",
                margin=dict(l=40, r=20, t=60, b=40),
                template=template,
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Monthly Bars":
        tab_bars, tab_dist = st.tabs(["Bars", "Distribution"])
        with tab_bars:
            dfb = port.to_frame("Monthly Return").reset_index(names="Month")
            fig_bar = px.bar(
                dfb,
                x="Month",
                y="Monthly Return",
                title="Portfolio Monthly Returns",
                color="Monthly Return",
                color_continuous_scale="RdYlGn",
                template=template,
            )
            fig_bar.update_yaxes(tickformat=".0%")
            fig_bar.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_bar, use_container_width=True)
        with tab_dist:
            render_distribution_panel(st, port, template)

    elif chart == "Correlation":
        filt = panel.copy()
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
                template=template,
            )
            fig_corr.update_layout(width=size, height=size, margin=dict(l=60, r=20, t=60, b=60))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select ≥2 managers with overlapping data to display correlation.")

    elif chart == "Year×Month":
        dfym = port.to_frame("ret").copy()
        dfym["Year"] = dfym.index.year
        dfym["Month"] = dfym.index.strftime("%b")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        piv = dfym.pivot(index="Year", columns="Month", values="ret")
        piv = piv[[c for c in month_order if c in piv.columns]].sort_index()
        ytd = dfym.groupby("Year")["ret"].apply(lambda x: (1.0 + x).prod() - 1.0).reindex(piv.index)
        piv["YTD"] = ytd
        final_cols = [c for c in month_order if c in piv.columns] + ["YTD"]
        piv = piv[final_cols]
        if piv.empty:
            st.info("No data to build Year × Month table.")
        else:
            month_cols = [c for c in month_order if c in piv.columns]
            years = piv.index.tolist()
            z_months = piv[month_cols].values if month_cols else None
            z_ytd = piv[["YTD"]].values

            def _sym_range(arr: np.ndarray) -> tuple[float, float]:
                finite = np.asarray(arr, dtype=float)
                finite = finite[np.isfinite(finite)]
                if finite.size == 0:
                    return (-1.0, 1.0)
                m = float(np.nanmax(np.abs(finite)))
                if m == 0 or not np.isfinite(m):
                    m = 1.0
                return (-m, m)

            zmin_m, zmax_m = _sym_range(z_months) if month_cols else (-1.0, 1.0)
            zmin_y, zmax_y = _sym_range(z_ytd)

            month_weight = max(1, len(month_cols))
            ytd_weight = 1
            total = month_weight + ytd_weight
            fig_hm = make_subplots(
                rows=1,
                cols=2,
                shared_yaxes=True,
                horizontal_spacing=0.04,
                column_widths=[month_weight / total, ytd_weight / total],
            )
            if month_cols:
                text_m = [[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in piv[month_cols].values]
                fig_hm.add_trace(
                    go.Heatmap(
                        z=z_months,
                        x=month_cols,
                        y=years,
                        colorscale="RdYlGn",
                        zmin=zmin_m,
                        zmax=zmax_m,
                        hovertemplate="Year=%{y}<br>Month=%{x}<br>Return=%{z:.1%}<extra></extra>",
                        showscale=False,
                        text=text_m,
                        texttemplate="%{text}",
                        textfont=dict(size=11),
                    ),
                    row=1,
                    col=1,
                )
            else:
                fig_hm.add_trace(
                    go.Heatmap(z=[[None]], x=["-"], y=[years[0] if years else "-"], showscale=False),
                    row=1,
                    col=1,
                )
            text_y = [[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in piv[["YTD"]].values]
            fig_hm.add_trace(
                go.Heatmap(
                    z=z_ytd,
                    x=["YTD"],
                    y=years,
                    colorscale="RdYlGn",
                    zmin=zmin_y,
                    zmax=zmax_y,
                    hovertemplate="Year=%{y}<br>Column=%{x}<br>Return=%{z:.1%}<extra></extra>",
                    showscale=False,
                    text=text_y,
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                ),
                row=1,
                col=2,
            )
            fig_hm.update_layout(template=template, margin=dict(l=60, r=20, t=30, b=50))
            fig_hm.update_xaxes(title_text=None, row=1, col=1)
            fig_hm.update_xaxes(title_text=None, row=1, col=2)
            fig_hm.update_yaxes(title_text=None, autorange="reversed", row=1, col=1)
            try:
                d1 = fig_hm.layout.xaxis.domain[1]
                d2 = fig_hm.layout.xaxis2.domain[0]
                x0 = (d1 + d2) / 2.0 - 0.0015
                x1 = (d1 + d2) / 2.0 + 0.0015
                fig_hm.add_shape(
                    type="rect",
                    xref="paper",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    line=dict(width=0),
                    fillcolor="black",
                )
            except Exception:
                pass
            st.plotly_chart(fig_hm, use_container_width=True)

    # ---- Contributions & Actions ----
    st.subheader("Manager Contributions to Return and Risk")
    window = st.slider("Rolling window (months)", 12, 60, 36, 6, key="contrib_window")
    rc_df = overall_return_contrib(panel[chosen], w)
    trc_df, port_var = overall_risk_contrib(panel[chosen], w)

    st.markdown("**Overall (annualised)**")
    st.dataframe(
        rc_df.join(trc_df[["TRC%"]], how="left").style.format({"Mu(ann)": "{:.2%}", "RC": "{:.2%}", "RC%": "{:.1%}", "TRC%": "{:.1%}"})
    )

    roll = rolling_contrib(panel[chosen], w, window=window)
    if not roll.empty:
        trc_ts = roll.xs("TRC%", level=1)
        st.area_chart(trc_ts)
        rc_ts = roll.xs("RC%", level=1)
        st.area_chart(rc_ts)

    meta = st.session_state.get("_params", {})

    c_dl, c_share = st.columns([1, 1])
    with c_dl:
        if st.button("Download PDF report", type="primary"):
            try:
                pdf_bytes = build_pdf_detailed(port, panel[chosen], w, meta, rc_df, trc_df, roll)
            except Exception as e:
                if build_pdf_basic is not None:
                    pdf_bytes = build_pdf_basic(port, panel[chosen], w, meta)
                else:
                    raise e
            st.download_button("Save PDF", data=pdf_bytes, file_name="portfolio_report.pdf", mime="application/pdf")
    with c_share:
        st.button("Share / bookmark this setup", on_click=encode_state_to_query)
    st.caption("Click the button, then copy the URL from your browser.")

# ==============================================================
# Distribution analysis panel
# ==============================================================

def _summary_statistics_series(s: pd.Series, rf_monthly: float = 0.0, var_levels=(0.95, 0.99)) -> pd.DataFrame:
    s = s.dropna()
    if len(s) == 0:
        return pd.DataFrame()
    ann = 12.0
    mean_m = s.mean()
    std_m = s.std(ddof=1)
    sharpe_m = (mean_m - rf_monthly) / std_m if std_m > 0 else np.nan
    stats_dict = {
        "count_months": [s.count()],
        "mean_m": [mean_m],
        "std_m": [std_m],
        "skew": [s.skew()],
        "kurtosis": [s.kurtosis()],
        "%_positive": [(s > 0).mean()],
        "best_month": [s.max()],
        "best_month_date": [s.idxmax()],
        "worst_month": [s.min()],
        "worst_month_date": [s.idxmin()],
        "sharpe_m": [sharpe_m],
        "mean_ann": [mean_m * ann],
        "std_ann": [std_m * math.sqrt(ann)],
        "sharpe_ann": [sharpe_m * math.sqrt(ann) if np.isfinite(sharpe_m) else np.nan],
    }
    for q in var_levels:
        var_q = s.quantile(1 - q)
        es_q = s[s <= var_q].mean() if (s <= var_q).any() else np.nan
        stats_dict[f"VaR_{int(q*100)}"] = [var_q]
        stats_dict[f"ES_{int(q*100)}"] = [es_q]
    return pd.DataFrame(stats_dict)


def render_distribution_panel(st, monthly_portfolio: pd.Series, template: Optional[str] = None):
    st.subheader("Monthly Return Distribution")

    bins_choice = st.selectbox("Binning", ["Freedman–Diaconis", "Scott", "Sturges", "Fixed (choose N)"], index=0)
    nbins = None
    if bins_choice.startswith("Fixed"):
        nbins = st.slider("Number of bins", min_value=10, max_value=200, value=50, step=5)
    kde = st.checkbox("KDE overlay", value=True)
    rug = st.checkbox("Show rug", value=False)
    wins = st.checkbox("Winsorize 1% tails", value=False)

    s = monthly_portfolio.copy()
    if wins:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)

    min_d, max_d = s.index.min(), s.index.max()
    dr = st.slider("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    s = s.loc[dr[0]:dr[1]]

    data = s.dropna()

    # Histogram
    if bins_choice == "Freedman–Diaconis":
        bins = "fd"
    elif bins_choice == "Scott":
        bins = "scott"
    elif bins_choice == "Sturges":
        bins = "sturges"
    else:
        bins = nbins

    df_hist = pd.DataFrame({"ret": data, "sign": np.where(data >= 0, ">= 0", "< 0")})
    fig_h = px.histogram(
        df_hist,
        x="ret",
        color="sign",
        nbins=None if isinstance(bins, str) else bins,
        barmode="overlay",
        color_discrete_map={">= 0": "#2ca02c", "< 0": "#d62728"},
        template=template,
    )

    if kde and stats is not None and len(data) > 3:
        xs = np.linspace(float(data.min()), float(data.max()), 256)
        kde_est = stats.gaussian_kde(data)
        ys = kde_est(xs)
        # scale density to histogram height
        max_y = 0
        for t in fig_h.data:
            if getattr(t, "y", None) is not None:
                try:
                    max_y = max(max_y, float(np.nanmax(t.y)))
                except Exception:
                    pass
        scale = (max_y / ys.max()) if ys.max() > 0 else 1.0
        fig_h.add_trace(
            go.Scatter(x=xs, y=ys * scale, mode="lines", name="KDE", line=dict(color="#1f77b4", width=2))
        )

    if rug:
        fig_h.add_trace(
            go.Scatter(
                x=data,
                y=[0] * len(data),
                mode="markers",
                name="Rug",
                marker=dict(color="rgba(0,0,0,0.35)", size=6),
                hoverinfo="x",
            )
        )

    fig_h.update_layout(
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis_title="Monthly return",
        yaxis_title="Count",
    )
    fig_h.update_xaxes(tickformat=".1%")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig_h, use_container_width=True)
    with c2:
        stats_df = _summary_statistics_series(s)
        st.dataframe(stats_df.T.rename(columns={0: "value"}))
        st.download_button(
            "Download stats (CSV)", data=stats_df.to_csv(index=True), file_name="monthly_distribution_stats.csv", mime="text/csv"
        )

    c3, c4 = st.columns(2)
    with c3:
        fig_v = go.Figure()
        fig_v.add_trace(go.Violin(y=data, name="Violin", box_visible=True, meanline_visible=True))
        fig_v.add_trace(go.Box(y=data, name="Box", boxpoints="outliers"))
        fig_v.update_layout(showlegend=False, yaxis_title="Monthly return", template=template)
        fig_v.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_v, use_container_width=True)
    with c4:
        xs = np.sort(data.values)
        ys = np.arange(1, len(xs) + 1) / len(xs) if len(xs) else np.array([0])
        fig_e = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="ECDF"))
        fig_e.update_layout(xaxis_title="Monthly return", yaxis_title="Cumulative probability", template=template)
        fig_e.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_e, use_container_width=True)

    if stats is not None and len(data) > 0:
        osm, osr = stats.probplot(data.values, dist="norm", sparams=(), fit=False)
        fig_q = go.Figure(go.Scatter(x=osm, y=osr, mode="markers", name="Data"))
        lo, hi = (np.nanpercentile(osm, [1, 99]) if len(osm) > 1 else (-3, 3))
        fig_q.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="45°", line=dict(color="gray", dash="dash")))
        fig_q.update_layout(xaxis_title="Theoretical quantiles (Normal)", yaxis_title="Sample quantiles", template=template)
        fig_q.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_q, use_container_width=True)

    st.download_button(
        "Download monthly returns (CSV)",
        data=pd.DataFrame({"ret": s.dropna()}).to_csv(index=True),
        file_name="monthly_returns.csv",
        mime="text/csv",
    )

# ==============================================================
# Detailed PDF builder (Matplotlib/Seaborn + PyMuPDF)
# ==============================================================

def _mpl_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _plot_monthly_bars_png(port: pd.Series) -> Optional[bytes]:
    if plt is None:
        return None
    s = port.dropna()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = np.where(s >= 0, "#2ca02c", "#d62728")
    ax.bar(s.index, s.values, color=colors)
    ax.set_title("Portfolio Monthly Returns")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
    fig.autofmt_xdate()
    return _mpl_png_bytes(fig)


def _plot_cum_and_dd_png(port: pd.Series) -> tuple[Optional[bytes], Optional[bytes]]:
    if plt is None:
        return None, None
    s = port.dropna()
    cum = (1 + s).cumprod()
    fig1, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(cum.index, cum.values, color="black")
    ax1.set_title("Cumulative Growth of £1")
    ax1.set_ylabel("Value (£)")
    fig1.autofmt_xdate()
    p1 = _mpl_png_bytes(fig1)

    peak = cum.cummax()
    dd = cum / peak - 1.0
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    ax2.plot(dd.index, dd.values, color="#d62728")
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
    fig2.autofmt_xdate()
    p2 = _mpl_png_bytes(fig2)
    return p1, p2


def _plot_rolling_png(port: pd.Series) -> tuple[Optional[bytes], Optional[bytes]]:
    if plt is None:
        return None, None
    s = port.dropna()
    rret = (1 + s).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0
    rvol = s.rolling(12, min_periods=12).std(ddof=0) * math.sqrt(12)
    p1 = p2 = None
    if not rret.dropna().empty:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(rret.index, rret.values, color="#1f77b4")
        ax.set_title("12M Rolling Return")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
        fig.autofmt_xdate()
        p1 = _mpl_png_bytes(fig)
    if not rvol.dropna().empty:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(rvol.index, rvol.values, color="#ff7f0e")
        ax.set_title("12M Rolling Volatility (ann.)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
        fig.autofmt_xdate()
        p2 = _mpl_png_bytes(fig)
    return p1, p2


def _plot_distribution_pngs(port: pd.Series) -> tuple[Optional[bytes], Optional[bytes], Optional[bytes]]:
    if plt is None or sns is None:
        return None, None, None
    s = port.dropna()
    # Hist + KDE
    fig1, ax1 = plt.subplots(figsize=(6.5, 4))
    sns.histplot(s, bins="fd", kde=(stats is not None), ax=ax1, color="#1f77b4")
    ax1.set_title("Distribution (Hist + KDE)")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1%}"))
    p1 = _mpl_png_bytes(fig1)
    # Violin/Box
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    sns.violinplot(y=s, inner="box", color="#9edae5", ax=ax2)
    ax2.set_title("Violin + Box")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1%}"))
    p2 = _mpl_png_bytes(fig2)
    # ECDF
    fig3, ax3 = plt.subplots(figsize=(6.5, 4))
    xs = np.sort(s.values)
    ys = np.arange(1, len(xs) + 1) / len(xs) if len(xs) else np.array([0])
    ax3.plot(xs, ys)
    ax3.set_title("ECDF")
    ax3.set_xlabel("Monthly return")
    ax3.set_ylabel("Cumulative probability")
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.1%}"))
    p3 = _mpl_png_bytes(fig3)
    return p1, p2, p3


def _plot_correlation_png(panel: pd.DataFrame) -> Optional[bytes]:
    if plt is None or sns is None:
        return None
    if panel.shape[1] < 2:
        return None
    corr = panel.corr(min_periods=3)
    fig, ax = plt.subplots(figsize=(6 + 0.35 * panel.shape[1], 5))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation (monthly returns)")
    return _mpl_png_bytes(fig)


def _plot_year_month_png(port: pd.Series) -> Optional[bytes]:
    if plt is None or sns is None:
        return None
    s = port.dropna()
    if s.empty:
        return None
    df = s.to_frame("ret").copy()
    df["Year"] = df.index.year
    df["Month"] = df.index.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    piv = df.pivot(index="Year", columns="Month", values="ret")
    piv = piv[[c for c in month_order if c in piv.columns]].sort_index()
    if piv.empty:
        return None
    ytd = df.groupby("Year")["ret"].apply(lambda x: (1 + x).prod() - 1.0).reindex(piv.index)
    piv["YTD"] = ytd
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.42 * piv.shape[0])))
    sns.heatmap(piv, cmap="RdYlGn", center=0.0, fmt=".1%", annot=True, cbar=True, ax=ax)
    ax.set_title("Year × Month (incl. YTD)")
    return _mpl_png_bytes(fig)


def build_pdf_detailed(
    port: pd.Series,
    panel_sel: pd.DataFrame,
    weights: pd.Series,
    meta: dict,
    rc_df: Optional[pd.DataFrame] = None,
    trc_df: Optional[pd.DataFrame] = None,
    roll: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Compose a multi-page PDF mirroring the Explorer analyses using PyMuPDF.
    Falls back to basic builder if PyMuPDF is unavailable.
    """
    if fitz is None or plt is None:
        if build_pdf_basic is not None:
            return build_pdf_basic(port, panel_sel, weights, meta)
        raise RuntimeError("PyMuPDF/Matplotlib not available to build the detailed PDF.")

    doc = fitz.open()

    # Cover / parameters
    page = doc.new_page()
    rect = page.rect
    title = "Portfolio Explorer — Detailed Report"
    subtitle = f"Period: {port.index.min().date()} → {port.index.max().date()}"
    params = (
        f"FX: {meta.get('fx_mode','-')} | Hedge ratio: {meta.get('hedge_ratio','-')}\n"
        f"GBP cash: {meta.get('gbp_cash_ann','-')} | USD cash: {meta.get('usd_cash_ann','-')}"
    )
    page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 90), title, fontsize=20, fontname="helv")
    page.insert_textbox(fitz.Rect(36, 76, rect.width - 36, 120), subtitle, fontsize=11, fontname="helv")
    page.insert_textbox(fitz.Rect(36, 96, rect.width - 36, 150), params, fontsize=10, fontname="helv")
    wt_txt = (weights.round(4).to_string() if weights is not None else "-")
    page.insert_textbox(
        fitz.Rect(36, 150, rect.width - 36, rect.height - 36), "Weights (normalised):\n" + wt_txt, fontsize=10
    )

    # Summary page
    summ = summarize(port)
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 70), "Summary", fontsize=18, fontname="helv")
    txt_lines = []
    for k in ["Ann. Return", "Ann. Vol", "Sharpe (rf=0)", "Max Drawdown", "Calmar"]:
        v = summ.get(k)
        if v is None:
            continue
        if any(x in k for x in ["Return", "Vol", "Drawdown"]):
            txt_lines.append(f"{k}: {v:.2%}")
        elif "Sharpe" in k or "Calmar" in k:
            txt_lines.append(f"{k}: {v:.2f}")
        else:
            txt_lines.append(f"{k}: {v}")
    page.insert_textbox(fitz.Rect(36, 76, 330, 320), "\n".join(txt_lines), fontsize=10)
    png_bars = _plot_monthly_bars_png(port)
    if png_bars:
        page.insert_image(fitz.Rect(350, 76, rect.width - 36, 300), stream=png_bars)

    # Cum & DD page
    png_cum, png_dd = _plot_cum_and_dd_png(port)
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 70), "Cumulative & Drawdown", fontsize=18, fontname="helv")
    if png_cum:
        page.insert_image(fitz.Rect(36, 80, rect.width - 36, 280), stream=png_cum)
    if png_dd:
        page.insert_image(fitz.Rect(36, 300, rect.width - 36, 520), stream=png_dd)

    # Rolling page
    png_rr, png_rv = _plot_rolling_png(port)
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 70), "Rolling Metrics", fontsize=18, fontname="helv")
    if png_rr:
        page.insert_image(fitz.Rect(36, 80, rect.width - 36, 280), stream=png_rr)
    if png_rv:
        page.insert_image(fitz.Rect(36, 300, rect.width - 36, 520), stream=png_rv)

    # Distribution page
    p1, p2, p3 = _plot_distribution_pngs(port)
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 70), "Distribution", fontsize=18, fontname="helv")
    if p1:
        page.insert_image(fitz.Rect(36, 80, rect.width / 2 - 18, 300), stream=p1)
    if p2:
        page.insert_image(fitz.Rect(rect.width / 2 + 18, 80, rect.width - 36, 300), stream=p2)
    if p3:
        page.insert_image(fitz.Rect(36, 320, rect.width - 36, 540), stream=p3)

    # Correlation page
    png_corr = _plot_correlation_png(panel_sel)
    if png_corr:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 70), "Correlation", fontsize=18, fontname="helv")
        page.insert_image(fitz.Rect(36, 80, rect.width - 36, 540), stream=png_corr)

    # Year×Month page
    png_ym = _plot_year_month_png(port)
    if png_ym:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 70), "Year × Month", fontsize=18, fontname="helv")
        page.insert_image(fitz.Rect(36, 80, rect.width - 36, 540), stream=png_ym)

    # Contributions page(s)
    if rc_df is not None and not rc_df.empty:
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(36, 36, rect.width - 36, 70), "Return Contribution (annualised)", fontsize=18, fontname="helv"
        )
        # dump as text table for portability
        rc_txt = rc_df.round(4).to_string()
        page.insert_textbox(fitz.Rect(36, 80, rect.width - 36, 540), rc_txt, fontsize=9)
    if trc_df is not None and not trc_df.empty:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(36, 36, rect.width - 36, 70), "Total Risk Contribution (TRC%)", fontsize=18, fontname="helv")
        trc_txt = trc_df.round(4).to_string()
        page.insert_textbox(fitz.Rect(36, 80, rect.width - 36, 540), trc_txt, fontsize=9)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
