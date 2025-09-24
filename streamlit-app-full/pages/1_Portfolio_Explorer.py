
# 1_Portfolio_Explorer.py
# Minimal change version:
# - Leaves everything else untouched (contributions, PDF, layout).
# - Keeps weights sliders at 0..1.
# - Adds/fixes "Monthly Bars → Distribution" with helper functions defined BEFORE use.

import math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.hedging import build_hedging_inputs, build_panel_for_selection
from src.metrics import summarize, compute_drawdown
from src.contrib import overall_return_contrib, overall_risk_contrib, rolling_contrib
from src.report import build_pdf
from src.state import load_state_from_query, encode_state_to_query, apply_theme, get_plotly_template


# ---------------------------------------------------------------------------
# Distribution helpers (ADDED) — minimal, self-contained, no other changes.
# ---------------------------------------------------------------------------

def _summary_statistics_series(
    s: pd.Series,
    rf_monthly: float = 0.0,
    var_levels=(0.95, 0.99),
) -> pd.DataFrame:
    """
    Compact summary stats table for a monthly return series.
    Works without SciPy; VaR/ES are empirical (quantile + tail mean).
    """
    s = s.dropna()
    if s.empty:
        return pd.DataFrame()

    ann = 12.0
    mean_m = s.mean()
    std_m  = s.std(ddof=1)
    sharpe_m = (mean_m - rf_monthly) / std_m if std_m > 0 else np.nan

    out = {
        "count_months": [int(s.count())],
        "mean_m":       [mean_m],
        "std_m":        [std_m],
        "skew":         [s.skew()],
        "kurtosis":     [s.kurtosis()],
        "%_positive":   [(s > 0).mean()],
        "best_month":   [s.max()],
        "best_month_date":  [s.idxmax()],
        "worst_month":      [s.min()],
        "worst_month_date": [s.idxmin()],
        "sharpe_m":     [sharpe_m],
        "mean_ann":     [mean_m * ann],
        "std_ann":      [std_m * np.sqrt(ann)],
        "sharpe_ann":   [sharpe_m * np.sqrt(ann) if np.isfinite(sharpe_m) else np.nan],
    }
    for q in var_levels:
        var_q = s.quantile(1 - q)                   # e.g., 5th pct for 95% VaR
        es_q  = s[s <= var_q].mean() if (s <= var_q).any() else np.nan
        out[f"VaR_{int(q*100)}"] = [var_q]
        out[f"ES_{int(q*100)}"]  = [es_q]

    return pd.DataFrame(out)


def render_distribution_panel(st, monthly_portfolio: pd.Series, template=None):
    """
    Renders Distribution sub-tab for Monthly Bars.
    - Histogram (sign-coloured), optional KDE (if SciPy available)
    - Violin + Box
    - ECDF
    - Q–Q (if SciPy available)
    - Summary stats + CSV downloads
    """
    # Defer SciPy import so KDE/QQ gracefully disable if not installed
    try:
        from scipy import stats as _stats
    except Exception:
        _stats = None

    st.subheader("Monthly Return Distribution")

    # Controls
    bins_choice = st.selectbox(
        "Binning",
        ["Freedman–Diaconis", "Scott", "Sturges", "Fixed (choose N)"],
        index=0,
    )
    nbins = None
    if bins_choice.startswith("Fixed"):
        nbins = st.slider("Number of bins", min_value=10, max_value=200, value=50, step=5)
    kde  = st.checkbox("KDE overlay", value=bool(_stats))   # auto-enable if SciPy is available
    rug  = st.checkbox("Show rug", value=False)
    wins = st.checkbox("Winsorize 1% tails", value=False)

    # Series prep
    s = monthly_portfolio.copy()
    if wins and not s.dropna().empty:
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)

    # Date range filter
    if not s.dropna().empty:
        min_d, max_d = s.index.min(), s.index.max()
        dr = st.slider("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        s = s.loc[dr[0]:dr[1]]

    data = s.dropna()
    if data.empty:
        st.info("No data in the selected range.")
        return

    # Choose histogram binning
    if bins_choice == "Freedman–Diaconis":
        bins = "fd"
    elif bins_choice == "Scott":
        bins = "scott"
    elif bins_choice == "Sturges":
        bins = "sturges"
    else:
        bins = nbins

    # Histogram (sign-coloured)
    df_hist = pd.DataFrame({"ret": data, "sign": np.where(data >= 0, ">= 0", "< 0")})
    fig_h = px.histogram(
        df_hist, x="ret", color="sign",
        nbins=None if isinstance(bins, str) else bins,
        barmode="overlay",
        color_discrete_map={">= 0": "#2ca02c", "< 0": "#d62728"},
        template=template,
    )

    # KDE (scaled to histogram height) — only if SciPy available
    if kde and _stats is not None and len(data) > 3:
        xs = np.linspace(float(data.min()), float(data.max()), 256)
        kde_est = _stats.gaussian_kde(data.values)
        ys = kde_est(xs)

        # scale density to histogram y-range
        max_y = 0.0
        for tr in fig_h.data:
            y = getattr(tr, "y", None)
            if y is not None:
                try:
                    max_y = max(max_y, float(np.nanmax(y)))
                except Exception:
                    pass
        scale = (max_y / float(np.nanmax(ys))) if np.nanmax(ys) > 0 else 1.0
        fig_h.add_trace(
            go.Scatter(x=xs, y=ys * scale, mode="lines", name="KDE",
                       line=dict(color="#1f77b4", width=2))
        )

    if rug:
        fig_h.add_trace(
            go.Scatter(
                x=data, y=[0]*len(data), mode="markers", name="Rug",
                marker=dict(color="rgba(0,0,0,0.35)", size=6), hoverinfo="x"
            )
        )

    fig_h.update_layout(
        bargap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis_title="Monthly return", yaxis_title="Count"
    )
    fig_h.update_xaxes(tickformat=".1%")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig_h, use_container_width=True)

    # Stats + CSV
    stats_df = _summary_statistics_series(s)
    with c2:
        st.dataframe(stats_df.T.rename(columns={0: "value"}))
        st.download_button(
            "Download stats (CSV)",
            data=stats_df.to_csv(index=True),
            file_name="monthly_distribution_stats.csv",
            mime="text/csv",
        )

    # Secondary charts
    c3, c4 = st.columns(2)

    # Violin + Box
    with c3:
        fig_v = go.Figure()
        fig_v.add_trace(go.Violin(y=data, name="Violin", box_visible=True, meanline_visible=True))
        fig_v.add_trace(go.Box(y=data, name="Box", boxpoints="outliers"))
        fig_v.update_layout(showlegend=False, yaxis_title="Monthly return", template=template)
        fig_v.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_v, use_container_width=True)

    # ECDF and Q–Q
    with c4:
        # ECDF
        xs = np.sort(data.values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        fig_e = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="ECDF"))
        fig_e.update_layout(xaxis_title="Monthly return", yaxis_title="Cumulative probability", template=template)
        fig_e.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_e, use_container_width=True)

        # Q–Q (only if SciPy available)
        if _stats is not None and len(data) > 0:
            osm, osr = _stats.probplot(data.values, dist="norm", fit=False)
            fig_q = go.Figure(go.Scatter(x=osm, y=osr, mode="markers", name="Q–Q"))
            lo, hi = (np.nanpercentile(osm, [1, 99]) if len(osm) > 1 else (-3, 3))
            fig_q.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                       name="45°", line=dict(color="gray", dash="dash")))
            fig_q.update_layout(
                xaxis_title="Theoretical quantiles (Normal)",
                yaxis_title="Sample quantiles",
                template=template
            )
            fig_q.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_q, use_container_width=True)

    # Underlying series CSV
    st.download_button(
        "Download monthly returns (CSV)",
        data=pd.DataFrame({"ret": s.dropna()}).to_csv(index=True),
        file_name="monthly_returns.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------------------------------
# Original app code (unchanged aside from weights slider 0..1 and Distribution tab)
# ------------------------------------------------------------------------------------

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
        # CHANGED: non-negative slider 0..1 (everything else untouched)
        weights[m] = st.sidebar.slider(m, 0.0, 1.0, float(w0), 0.01)

all_months = pd.Index(sorted(set().union(*[s.index for s in man_local_m.values()])))
earliest_date, latest_date = all_months.min().date(), all_months.max().date()

default_start = max(pd.to_datetime("2016-01-01").date(), earliest_date)
start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
    min_value=earliest_date,
    max_value=latest_date
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
        start_ts=start_ts
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
    st.session_state["_params"] = dict(fx_mode=fx_mode, hedge_ratio=hedge_ratio, gbp_cash_ann=gbp_cash_ann, usd_cash_ann=usd_cash_ann)

chart = st.radio("Chart", options=[
    "Summary","Cumulative","Drawdown","12M Return","12M Vol","Monthly Bars","Correlation","Year×Month"
], index=0, horizontal=True)

panel = st.session_state.get("_panel")
port = st.session_state.get("_port")
chosen = st.session_state.get("_chosen", [])
w = st.session_state.get("_weights")

if port is None or panel is None or not chosen:
    st.info("Select managers, set weights & hedging, pick a start date, then click **Run portfolio analytics**.")
else:
    st.markdown(f"**Period:** {port.index.min().date()} → {port.index.max().date()}")

    if chart == "Summary":
        stats = summarize(port)
        df_stats = pd.DataFrame(stats, index=[0]).T.rename(columns={0: "Value"})
        st.dataframe(df_stats.style.format({
            "Ann. Return":"{:.2%}",
            "Ann. Vol":"{:.2%}",
            "Max Drawdown":"{:.2%}",
            "Sharpe (rf=0)":"{:.2f}",
            "Calmar":"{:.2f}"
        }))
        st.markdown("**Active Weights**")
        st.dataframe(pd.DataFrame({"Manager": w.index, "Weight": w.values}).style.format({"Weight":"{:.2%}"}))

    elif chart == "Cumulative":
        fig = go.Figure()
        cum_port = (1.0 + port).cumprod()
        fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, name="Portfolio",
                                 line=dict(width=3, color="black")))
        for m in chosen:
            s = panel[m].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=(1+s).cumprod(), name=m, line=dict(width=1), opacity=0.5))
        fig.update_layout(title="Cumulative Growth of £1", hovermode="x unified",
                          legend=dict(orientation="h"), yaxis_title="Value (£)", xaxis_title="Date",
                          margin=dict(l=40, r=20, t=60, b=40), template=template)
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "Drawdown":
        cum_port = (1.0 + port).cumprod()
        dd = compute_drawdown(cum_port)
        y_min = min(-1.0, float(dd.min()) * 1.05) if np.isfinite(dd.min()) else -1.0
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", line=dict(color="#d62728", width=2)))
        fig.update_layout(title="Portfolio Drawdown", hovermode="x unified",
                          xaxis_title="Date", yaxis_title="Drawdown",
                          margin=dict(l=40, r=20, t=60, b=40), template=template)
        fig.update_yaxes(tickformat=".0%", range=[y_min, 0])
        st.plotly_chart(fig, use_container_width=True)

    elif chart == "12M Return":
        roll12_ret = (1.0 + port).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0
        if roll12_ret.dropna().empty:
            st.info("Not enough data (need ≥12 months).")
        else:
            fig = go.Figure(go.Scatter(x=roll12_ret.index, y=roll12_ret.values,
                                       name="12M Rolling Return", line=dict(color="#1f77b4", width=2)))
            fig.update_layout(title="12‑month Rolling Return", hovermode="x unified",
                              xaxis_title="Date", yaxis_title="Return (12M)",
                              margin=dict(l=40, r=20, t=60, b=40), template=template)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "12M Vol":
        roll12_vol = port.rolling(12, min_periods=12).std(ddof=0) * math.sqrt(12)
        if roll12_vol.dropna().empty:
            st.info("Not enough data (need ≥12 months).")
        else:
            fig = go.Figure(go.Scatter(x=roll12_vol.index, y=roll12_vol.values,
                                       name="12M Rolling Vol (ann.)", line=dict(color="#ff7f0e", width=2)))
            fig.update_layout(title="12‑month Rolling Volatility (Annualised)",
                              hovermode="x unified", xaxis_title="Date", yaxis_title="Volatility (ann.)",
                              margin=dict(l=40, r=20, t=60, b=40), template=template)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Monthly Bars":
        # Bars + Distribution (ADDED: Distribution sub-tab call only; Bars unchanged)
        tab_bars, tab_dist = st.tabs(["Bars", "Distribution"])
        with tab_bars:
            dfb = port.to_frame("Monthly Return").reset_index(names="Month")
            fig_bar = px.bar(dfb, x="Month", y="Monthly Return", title="Portfolio Monthly Returns",
                             color="Monthly Return", color_continuous_scale="RdYlGn", template=template)
            fig_bar.update_yaxes(tickformat=".0%")
            fig_bar.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_bar, use_container_width=True)
        with tab_dist:
            # new helper lives above; no other code changed
            render_distribution_panel(st, port, template)

    elif chart == "Correlation":
        filt = panel.copy()
        valid_cols = [c for c in filt.columns if filt[c].notna().any()]
        filt = filt[valid_cols]
        if len(valid_cols) >= 2:
            corr = filt.corr(min_periods=3).dropna(how="all").dropna(how="all", axis=1)
            size = max(420, 70 * corr.shape[0])
            fig_corr = px.imshow(corr.round(2), text_auto=True,
                                 color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                 title="Correlation (monthly returns)", template=template)
            fig_corr.update_layout(width=size, height=size, margin=dict(l=60, r=20, t=60, b=60))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select ≥2 managers with overlapping data to display correlation.")

    elif chart == "Year×Month":
        # Build Year×Month table and add YTD (unchanged logic)
        dfym = port.to_frame("ret").copy()
        dfym["Year"] = dfym.index.year
        dfym["Month"] = dfym.index.strftime("%b")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        piv = dfym.pivot(index="Year", columns="Month", values="ret")
        piv = piv[[c for c in month_order if c in piv.columns]].sort_index()
        # YTD (compounded across months available in that year)
        ytd = (
            dfym.groupby("Year")["ret"]
            .apply(lambda x: (1.0 + x).prod() - 1.0)
            .reindex(piv.index)
        )
        piv["YTD"] = ytd
        final_cols = [c for c in month_order if c in piv.columns] + ["YTD"]
        piv = piv[final_cols]
        if piv.empty:
            st.info("No data to build Year × Month table.")
        else:
            # Separate data for monthly columns vs YTD (distinct heatmaps / color scales)
            month_cols = [c for c in month_order if c in piv.columns]
            years = piv.index.tolist()
            z_months = piv[month_cols].values if month_cols else None
            z_ytd = piv[["YTD"]].values
            import numpy as _np
            def _sym_range(arr):
                finite = _np.asarray(arr, dtype=float)
                finite = finite[_np.isfinite(finite)]
                if finite.size == 0:
                    return (-1.0, 1.0)
                m = float(_np.nanmax(_np.abs(finite)))
                if m == 0 or not _np.isfinite(m):
                    m = 1.0
                return (-m, m)
            zmin_m, zmax_m = _sym_range(z_months) if month_cols else (-1.0, 1.0)
            zmin_y, zmax_y = _sym_range(z_ytd)
            # Two subplots, no subplot titles
            month_weight = max(1, len(month_cols))
            ytd_weight = 1
            total = month_weight + ytd_weight
            fig_hm = make_subplots(
                rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                column_widths=[month_weight/total, ytd_weight/total]
            )
            # Monthly heatmap (legend/scale hidden)
            if month_cols:
                text_m = [[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in piv[month_cols].values]
                fig_hm.add_trace(
                    go.Heatmap(
                        z=z_months, x=month_cols, y=years, colorscale='RdYlGn',
                        zmin=zmin_m, zmax=zmax_m,
                        hovertemplate="Year=%{y}<br>Month=%{x}<br>Return=%{z:.1%}<extra></extra>",
                        showscale=False, text=text_m, texttemplate="%{text}", textfont=dict(size=11),
                    ), row=1, col=1
                )
            else:
                fig_hm.add_trace(
                    go.Heatmap(z=[[None]], x=["-"], y=[years[0] if years else "-"], showscale=False),
                    row=1, col=1
                )
            # YTD heatmap (legend/scale hidden)
            text_y = [[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in piv[["YTD"]].values]
            fig_hm.add_trace(
                go.Heatmap(
                    z=z_ytd, x=["YTD"], y=years, colorscale='RdYlGn',
                    zmin=zmin_y, zmax=zmax_y,
                    hovertemplate="Year=%{y}<br>Column=%{x}<br>Return=%{z:.1%}<extra></extra>",
                    showscale=False, text=text_y, texttemplate="%{text}", textfont=dict(size=11),
                ), row=1, col=2
            )
            # Minimal layout (no title)
            fig_hm.update_layout(template=template, margin=dict(l=60, r=20, t=30, b=50))
            # Remove axis titles (keep ticks/labels only)
            fig_hm.update_xaxes(title_text=None, row=1, col=1)
            fig_hm.update_xaxes(title_text=None, row=1, col=2)
            fig_hm.update_yaxes(title_text=None, autorange='reversed', row=1, col=1)
            # Thick vertical border between panels
            try:
                d1 = fig_hm.layout.xaxis.domain[1]  # right edge of first subplot
                d2 = fig_hm.layout.xaxis2.domain[0] # left edge of second subplot
                x0 = (d1 + d2)/2.0 - 0.0015
                x1 = (d1 + d2)/2.0 + 0.0015
                fig_hm.add_shape(
                    type='rect', xref='paper', yref='paper', x0=x0, x1=x1, y0=0, y1=1,
                    line=dict(width=0), fillcolor='black'
                )
            except Exception:
                pass
            st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Manager Contributions to Return and Risk")
    window = st.slider("Rolling window (months)", 12, 60, 36, 6, key="contrib_window")
    rc_df = overall_return_contrib(panel[chosen], w)
    trc_df, port_var = overall_risk_contrib(panel[chosen], w)

    st.markdown("**Overall (annualised)**")
    st.dataframe(
        rc_df.join(trc_df[["TRC%"]], how="left")
        .style.format({"Mu(ann)":"{:.2%}","RC":"{:.2%}","RC%":"{:.1%}","TRC%":"{:.1%}"})
    )

    roll = rolling_contrib(panel[chosen], w, window=window)
    if not roll.empty:
        trc_ts = roll.xs("TRC%", level=1)
        st.area_chart(trc_ts)
        rc_ts = roll.xs("RC%", level=1)
        st.area_chart(rc_ts)

    meta = st.session_state.get("_params", {})
    c_dl, c_share = st.columns([1,1])
    with c_dl:
        if st.button("Download PDF report", type="primary"):
            # PDF flow UNCHANGED — uses your existing builder
            pdf_bytes = build_pdf(port, panel[chosen], w, meta)
            st.download_button("Save PDF", data=pdf_bytes, file_name="portfolio_report.pdf", mime="application/pdf")
    with c_share:
        st.button("Share / bookmark this setup", on_click=encode_state_to_query)
    st.caption("Click the button, then copy the URL from your browser.")
