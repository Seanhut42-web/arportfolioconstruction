# 1_Portfolio_Explorer.py
# Minimal change version (v7 - hardened):
# - Keeps everything else untouched (contributions, layout, simplified Distribution).
# - Keeps weights sliders at 0..1.
# - Full PDF export: includes EVERYTHING shown in Portfolio Explorer tab via Plotly->PNG (kaleido) + PyMuPDF (fitz).
#   Falls back to existing build_pdf if deps missing.
# - NEW: Robustness against stale manager selections after data reload/start-date changes:
#        * Reload button clears manager selections and cached data.
#        * Align chosen managers & weights to available panel columns before use.

import math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st

from src.hedging import build_hedging_inputs, build_panel_for_selection
from src.metrics import summarize, compute_drawdown
from src.contrib import overall_return_contrib, overall_risk_contrib, rolling_contrib
from src.report import build_pdf  # fallback
from src.state import load_state_from_query, encode_state_to_query, apply_theme, get_plotly_template


# ---------------------------------------------------------------------------
# Distribution (SIMPLIFIED): descriptive table + histogram only
# ---------------------------------------------------------------------------

def _desc_table(s: pd.Series) -> pd.DataFrame:
    s = s.dropna()
    if s.empty:
        return pd.DataFrame()
    # Basic descriptive stats + a few extras
    desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).rename({
        'count':'count', 'mean':'mean', 'std':'std', 'min':'min', '5%':'p5', '25%':'p25',
        '50%':'p50', '75%':'p75', '95%':'p95', 'max':'max'
    })
    extras = {
        '%_positive': (s > 0).mean(),
        'best_month': s.max(),
        'best_month_date': s.idxmax(),
        'worst_month': s.min(),
        'worst_month_date': s.idxmin(),
        'ann_mean': s.mean() * 12.0,
        'ann_std': s.std(ddof=1) * (12.0 ** 0.5),
    }
    desc_df = pd.concat([desc, pd.Series(extras)])
    return desc_df.to_frame('value')


def render_distribution_panel(st, monthly_portfolio: pd.Series, template=None):
    s = monthly_portfolio.dropna()
    if s.empty:
        st.info("No monthly returns to display.")
        return

    st.subheader("Monthly Return Distribution (Simple)")

    # Controls (only bins)
    bins_mode = st.selectbox("Binning", ["Auto", "Fixed (choose N)"], index=0)
    nbins = None
    if bins_mode.startswith("Fixed"):
        nbins = st.slider("Number of bins", min_value=10, max_value=200, value=50, step=5)

    c1, c2 = st.columns([1, 2])

    with c1:
        desc_df = _desc_table(s)

        def _fmt_row(idx, val):
            try:
                if any(k in idx for k in ["mean", "std", "min", "max", "p", "%_positive", "ann_"]):
                    return f"{val:.2%}" if pd.notna(val) else ""
                return f"{val}" if not isinstance(val, (float, int)) else f"{val:.4f}"
            except Exception:
                return f"{val}"

        st.dataframe(
            desc_df.assign(display=desc_df.index).set_index('display')
                   .apply(lambda col: [_fmt_row(idx, val) for idx, val in zip(desc_df.index, col)], axis=0)
        )
        st.download_button(
            "Download table (CSV)", data=desc_df.to_csv(index=True),
            file_name="monthly_distribution_table.csv", mime="text/csv"
        )

    with c2:
        fig_h = px.histogram(x=s.values, nbins=nbins if nbins else None, template=template)
        fig_h.update_traces(marker_color="#1f77b4")
        fig_h.update_layout(title="Distribution (Histogram)", xaxis_title="Monthly return", yaxis_title="Count")
        fig_h.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_h, use_container_width=True)


# ---------------------------------------------------------------------------
# PDF: full report (everything from Portfolio Explorer tab)
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def _fig_to_png(fig, scale: float = 2.0) -> bytes:
    """Convert a Plotly figure to PNG using kaleido."""
    return pio.to_image(fig, format="png", scale=scale, engine="kaleido")


# ---- Figure builders mirroring on-screen charts ----

def _fig_cumulative(port: pd.Series, panel: pd.DataFrame, chosen: list[str], template: str):
    fig = go.Figure()
    cum_port = (1.0 + port).cumprod()
    fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, name="Portfolio",
                             line=dict(width=3, color="black")))
    for m in chosen:
        s = panel[m].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=(1 + s).cumprod(), name=m,
                                 line=dict(width=1), opacity=0.5))
    fig.update_layout(title="Cumulative Growth of £1", hovermode="x unified",
                      legend=dict(orientation="h"), yaxis_title="Value (£)", xaxis_title="Date",
                      margin=dict(l=40, r=20, t=60, b=40), template=template)
    return fig


def _fig_drawdown(port: pd.Series, template: str):
    cum_port = (1.0 + port).cumprod()
    dd = compute_drawdown(cum_port)
    y_min = min(-1.0, float(dd.min()) * 1.05) if np.isfinite(dd.min()) else -1.0
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                               line=dict(color="#d62728", width=2)))
    fig.update_layout(title="Portfolio Drawdown", hovermode="x unified",
                      xaxis_title="Date", yaxis_title="Drawdown",
                      margin=dict(l=40, r=20, t=60, b=40), template=template)
    fig.update_yaxes(tickformat=".0%", range=[y_min, 0])
    return fig


def _fig_roll12_return(port: pd.Series, template: str):
    roll12_ret = (1.0 + port).rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0
    fig = go.Figure(go.Scatter(x=roll12_ret.index, y=roll12_ret.values,
                               name="12M Rolling Return", line=dict(color="#1f77b4", width=2)))
    fig.update_layout(title="12‑month Rolling Return", hovermode="x unified",
                      xaxis_title="Date", yaxis_title="Return (12M)",
                      margin=dict(l=40, r=20, t=60, b=40), template=template)
    fig.update_yaxes(tickformat=".0%")
    return fig


def _fig_roll12_vol(port: pd.Series, template: str):
    roll12_vol = port.rolling(12, min_periods=12).std(ddof=0) * math.sqrt(12)
    fig = go.Figure(go.Scatter(x=roll12_vol.index, y=roll12_vol.values,
                               name="12M Rolling Vol (ann.)", line=dict(color="#ff7f0e", width=2)))
    fig.update_layout(title="12‑month Rolling Volatility (Annualised)",
                      hovermode="x unified", xaxis_title="Date", yaxis_title="Volatility (ann.)",
                      margin=dict(l=40, r=20, t=60, b=40), template=template)
    fig.update_yaxes(tickformat=".0%")
    return fig


def _fig_monthly_bars(port: pd.Series, template: str):
    dfb = port.to_frame("Monthly Return").reset_index(names="Month")
    fig = px.bar(dfb, x="Month", y="Monthly Return", title="Portfolio Monthly Returns",
                 color="Monthly Return", color_continuous_scale="RdYlGn", template=template)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    return fig


def _fig_distribution_hist(port: pd.Series, template: str, nbins: int | None = None):
    s = port.dropna()
    fig = px.histogram(x=s.values, nbins=nbins, template=template)
    fig.update_traces(marker_color="#1f77b4")
    fig.update_layout(title="Distribution (Histogram)", xaxis_title="Monthly return", yaxis_title="Count")
    fig.update_xaxes(tickformat=".1%")
    return fig


def _fig_correlation(panel: pd.DataFrame, chosen: list[str], template: str):
    filt = panel.copy()
    valid_cols = [c for c in filt.columns if filt[c].notna().any()]
    filt = filt[valid_cols]
    if len(filt.columns) < 2:
        return None
    corr = filt.corr(min_periods=3).dropna(how="all").dropna(how="all", axis=1)
    fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Correlation (monthly returns)", template=template)
    fig.update_layout(margin=dict(l=60, r=20, t=60, b=60))
    return fig


def _fig_year_month(port: pd.Series, template: str):
    dfym = port.to_frame("ret").copy()
    if dfym.empty:
        return None
    dfym["Year"] = dfym.index.year
    dfym["Month"] = dfym.index.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    piv = dfym.pivot(index="Year", columns="Month", values="ret")
    piv = piv[[c for c in month_order if c in piv.columns]].sort_index()
    if piv.empty:
        return None
    ytd = dfym.groupby("Year")["ret"].apply(lambda x: (1.0 + x).prod() - 1.0).reindex(piv.index)
    piv["YTD"] = ytd
    final_cols = [c for c in month_order if c in piv.columns] + ["YTD"]
    piv = piv[final_cols]
    fig = px.imshow(piv, color_continuous_scale="RdYlGn", origin="upper",
                    title="Year × Month (incl. YTD)", template=template, aspect="auto")
    # Symmetric color scale around zero
    try:
        v = piv.values.astype(float)
        m = float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else 1.0
        fig.update_coloraxes(cmin=-m, cmax=m)
    except Exception:
        pass
    fig.update_layout(margin=dict(l=60, r=20, t=60, b=40))
    fig.update_xaxes(side="top")
    return fig


def _fig_area_from_wide(df_wide: pd.DataFrame, title: str, template: str):
    # Plotly area expects long format
    if df_wide is None or df_wide.empty:
        return None
    tmp = df_wide.copy()
    tmp.index.name = "Date"
    long_df = tmp.reset_index().melt("Date", var_name="Manager", value_name="value")
    fig = px.area(long_df, x="Date", y="value", color="Manager", template=template, title=title)
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(legend=dict(orientation="h"))
    return fig


def _distribution_table_text(s: pd.Series) -> str:
    s = s.dropna()
    desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).rename({
        'count': 'count', 'mean': 'mean', 'std': 'std', 'min': 'min',
        '5%': 'p5', '25%': 'p25', '50%': 'p50', '75%': 'p75', '95%': 'p95', 'max': 'max'
    })
    extras = {
        '%_positive': (s > 0).mean(),
        'best_month': s.max(),
        'best_month_date': s.idxmax(),
        'worst_month': s.min(),
        'worst_month_date': s.idxmin(),
        'ann_mean': s.mean() * 12.0,
        'ann_std': s.std(ddof=1) * (12.0 ** 0.5),
    }
    df = pd.concat([desc, pd.Series(extras)]).to_frame("value")
    lines = []
    for idx, val in df["value"].items():
        if isinstance(val, (float, np.floating)):
            if any(k in idx for k in ["mean", "std", "min", "max", "p", "%_positive", "ann_"]):
                lines.append(f"{idx:>18}: {val: .2%}")
            else:
                lines.append(f"{idx:>18}: {val: .6f}")
        else:
            lines.append(f"{idx:>18}: {val}")
    return "\n".join(lines)


def build_pdf_full_report(
    port: pd.Series,
    panel_sel: pd.DataFrame,
    chosen: list[str],
    weights: pd.Series,
    meta: dict,
    template: str,
    nbins: int | None = None,
) -> bytes:
    """
    Compose a multi-page PDF covering everything the Portfolio Explorer tab shows.
    Requires: kaleido (for Plotly -> PNG) and PyMuPDF (fitz) for PDF assembly.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not available. Please `pip install pymupdf`.")

    figs = []
    # On-screen equivalents
    figs.append(_fig_monthly_bars(port, template))
    figs.append(_fig_cumulative(port, panel_sel, chosen, template))
    figs.append(_fig_drawdown(port, template))

    # Rolling charts if data exists
    if (1.0 + port).rolling(12, min_periods=12).apply(np.prod, raw=True).dropna().size:
        figs.append(_fig_roll12_return(port, template))
    if port.rolling(12, min_periods=12).std(ddof=0).dropna().size:
        figs.append(_fig_roll12_vol(port, template))

    # Distribution
    dist_text = _distribution_table_text(port)
    figs.append(_fig_distribution_hist(port, template, nbins=nbins))

    # Correlation & Year×Month
    corr_fig = _fig_correlation(panel_sel, chosen, template)
    if corr_fig is not None:
        figs.append(corr_fig)
    ym_fig = _fig_year_month(port, template)
    if ym_fig is not None:
        figs.append(ym_fig)

    # Contributions
    rc_df = overall_return_contrib(panel_sel, weights)
    trc_df, _ = overall_risk_contrib(panel_sel, weights)
    # Rolling contributions
    try:
        roll = rolling_contrib(panel_sel, weights, window=36)
        trc_ts = roll.xs("TRC%", level=1) if not roll.empty else None
        rc_ts = roll.xs("RC%", level=1) if not roll.empty else None
    except Exception:
        trc_ts = None
        rc_ts = None
    trc_fig = _fig_area_from_wide(trc_ts, "Rolling TRC% (window=36)", template)
    rc_fig = _fig_area_from_wide(rc_ts, "Rolling RC% (window=36)", template)
    if trc_fig is not None:
        figs.append(trc_fig)
    if rc_fig is not None:
        figs.append(rc_fig)

    # ---- Assemble PDF ----
    doc = fitz.open()
    margin = 36
    page_w, page_h = 595, 842  # A4 portrait in points

    # Cover page
    page = doc.new_page(width=page_w, height=page_h)
    title = "Portfolio Explorer — Full Report"
    period = f"Period: {port.index.min().date()} → {port.index.max().date()}"
    params = f"FX: {meta.get('fx_mode','-')} | Hedge ratio: {meta.get('hedge_ratio','-')}\n" \
             f"GBP cash: {meta.get('gbp_cash_ann','-')} | USD cash: {meta.get('usd_cash_ann','-')}"

    page.insert_textbox(fitz.Rect(margin, margin, page_w - margin, margin + 36), title, fontsize=20, fontname="helv")
    page.insert_textbox(fitz.Rect(margin, margin + 36, page_w - margin, margin + 60), period, fontsize=11, fontname="helv")
    page.insert_textbox(fitz.Rect(margin, margin + 60, page_w - margin, margin + 100), params, fontsize=10, fontname="helv")

    # Summary + Weights
    summ = summarize(port)

    def _fmt_summ(k, v):
        if v is None:
            return None
        if any(x in k for x in ["Return", "Vol", "Drawdown"]):
            return f"{k}: {v:.2%}"
        if "Sharpe" in k or "Calmar" in k:
            return f"{k}: {v:.2f}"
        return f"{k}: {v}"

    lines = []
    for k in ["Ann. Return", "Ann. Vol", "Sharpe (rf=0)", "Max Drawdown", "Calmar"]:
        if k in summ:
            sline = _fmt_summ(k, summ[k])
            if sline:
                lines.append(sline)
    text_summ = "\n".join(lines) if lines else "—"
    weights_text = (weights.round(4).to_string() if weights is not None and not weights.empty else "—")

    page.insert_textbox(fitz.Rect(margin, margin + 110, page_w / 2 - 6, page_h / 2 - 40),
                        "Summary\n\n" + text_summ, fontsize=10, fontname="helv")
    page.insert_textbox(fitz.Rect(page_w / 2 + 6, margin + 110, page_w - margin, page_h / 2 - 40),
                        "Active weights (normalised)\n\n" + weights_text, fontsize=10, fontname="helv")

    # Distribution table (text)
    page.insert_textbox(fitz.Rect(margin, page_h / 2 - 20, page_w - margin, page_h - margin),
                        "Distribution (descriptive stats)\n\n" + dist_text, fontsize=9, fontname="helv")

    # Charts: two per page
    slot_h = (page_h - 2 * margin - 20) / 2
    page = None
    for i, fig in enumerate(figs):
        img = _fig_to_png(fig, scale=2.0)
        if i % 2 == 0:
            page = doc.new_page(width=page_w, height=page_h)
            top = margin
        else:
            top = margin + slot_h + 20
        rect = fitz.Rect(margin, top, page_w - margin, top + slot_h)
        page.insert_image(rect, stream=img)

    # Contributions tables
    page = doc.new_page(width=page_w, height=page_h)
    page.insert_textbox(fitz.Rect(margin, margin, page_w - margin, margin + 22),
                        "Return Contribution (annualised)", fontsize=14, fontname="helv")
    page.insert_textbox(fitz.Rect(margin, margin + 26, page_w - margin, page_h / 2 - 6),
                        rc_df.round(4).to_string(), fontsize=9, fontname="helv")
    page.insert_textbox(fitz.Rect(margin, page_h / 2 + 6, page_w - margin, page_h / 2 + 28),
                        "Total Risk Contribution (TRC%)", fontsize=14, fontname="helv")
    page.insert_textbox(fitz.Rect(margin, page_h / 2 + 32, page_w - margin, page_h - margin),
                        trc_df.round(4).to_string(), fontsize=9, fontname="helv")

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ------------------------------------------------------------------------------------
# Original app code (unchanged aside from weights slider 0..1, Distribution tab & PDF button)
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

# ------------------ HARDENED RELOAD: clear stale selections & caches ------------------
if st.sidebar.button("Reload data (after changing Excel)"):
    load_inputs.clear()
    # Clear manager checkboxes and cached selection/weights to avoid stale names
    for k in list(st.session_state.keys()):
        if k.startswith("mgr_"):
            del st.session_state[k]
    for k in ["_panel", "_port", "_weights", "_chosen"]:
        st.session_state.pop(k, None)
    st.experimental_rerun()
# -------------------------------------------------------------------------------------

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
    "Summary", "Cumulative", "Drawdown", "12M Return", "12M Vol", "Monthly Bars", "Correlation", "Year×Month"
], index=0, horizontal=True)

panel = st.session_state.get("_panel")
port = st.session_state.get("_port")
chosen = st.session_state.get("_chosen", [])
w = st.session_state.get("_weights")

# ------------------ HARDEN: align chosen & weights to panel columns -------------------
if panel is not None:
    panel_cols = list(panel.columns)
    chosen_existing = [m for m in chosen if m in panel_cols]
    missing = [m for m in chosen if m not in panel_cols]
    if missing:
        st.info("Removed managers not present in the current dataset: " + ", ".join(missing))
    chosen = chosen_existing

    if w is not None:
        w = w.reindex(chosen).dropna()
        if not w.empty and w.sum() != 0:
            w = w / w.sum()

    # keep session state in sync (optional but tidy)
    st.session_state["_chosen"] = chosen
    st.session_state["_weights"] = w
# -------------------------------------------------------------------------------------

if port is None or panel is None or not chosen:
    st.info("Select managers, set weights & hedging, pick a start date, then click **Run portfolio analytics**.")
else:
    st.markdown(f"**Period:** {port.index.min().date()} → {port.index.max().date()}")

    if chart == "Summary":
        stats = summarize(port)
        df_stats = pd.DataFrame(stats, index=[0]).T.rename(columns={0: "Value"})
        st.dataframe(df_stats.style.format({
            "Ann. Return": "{:.2%}",
            "Ann. Vol": "{:.2%}",
            "Max Drawdown": "{:.2%}",
            "Sharpe (rf=0)": "{:.2f}",
            "Calmar": "{:.2f}"
        }))
        st.markdown("**Active Weights**")
        st.dataframe(pd.DataFrame({"Manager": w.index, "Weight": w.values}).style.format({"Weight": "{:.2%}"}))

    elif chart == "Cumulative":
        fig = go.Figure()
        cum_port = (1.0 + port).cumprod()
        fig.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values, name="Portfolio",
                                 line=dict(width=3, color="black")))
        for m in chosen:
            s = panel[m].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=(1 + s).cumprod(), name=m, line=dict(width=1), opacity=0.5))
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
        tab_bars, tab_dist = st.tabs(["Bars", "Distribution"])
        with tab_bars:
            dfb = port.to_frame("Monthly Return").reset_index(names="Month")
            fig_bar = px.bar(dfb, x="Month", y="Monthly Return", title="Portfolio Monthly Returns",
                             color="Monthly Return", color_continuous_scale="RdYlGn", template=template)
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
            fig_corr = px.imshow(corr.round(2), text_auto=True,
                                 color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                 title="Correlation (monthly returns)", template=template)
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
        ytd = (
            dfym.groupby("Year")["ret"].apply(lambda x: (1.0 + x).prod() - 1.0).reindex(piv.index)
        )
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
            month_weight = max(1, len(month_cols))
            ytd_weight = 1
            total = month_weight + ytd_weight
            fig_hm = make_subplots(
                rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                column_widths=[month_weight / total, ytd_weight / total]
            )
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
            text_y = [[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in piv[["YTD"]].values]
            fig_hm.add_trace(
                go.Heatmap(
                    z=z_ytd, x=["YTD"], y=years, colorscale='RdYlGn',
                    zmin=zmin_y, zmax=zmax_y,
                    hovertemplate="Year=%{y}<br>Column=%{x}<br>Return=%{z:.1%}<extra></extra>",
                    showscale=False, text=text_y, texttemplate="%{text}", textfont=dict(size=11),
                ), row=1, col=2
            )
            fig_hm.update_layout(template=template, margin=dict(l=60, r=20, t=30, b=50))
            fig_hm.update_xaxes(title_text=None, row=1, col=1)
            fig_hm.update_xaxes(title_text=None, row=1, col=2)
            fig_hm.update_yaxes(title_text=None, autorange='reversed', row=1, col=1)
            try:
                d1 = fig_hm.layout.xaxis.domain[1]
                d2 = fig_hm.layout.xaxis2.domain[0]
                x0 = (d1 + d2) / 2.0 - 0.0015
                x1 = (d1 + d2) / 2.0 + 0.0015
                fig_hm.add_shape(type='rect', xref='paper', yref='paper', x0=x0, x1=x1, y0=0, y1=1,
                                 line=dict(width=0), fillcolor='black')
            except Exception:
                pass
            st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("Manager Contributions to Return and Risk")
    window = st.slider("Rolling window (months)", 12, 60, 36, 6, key="contrib_window")

    # Additional guard (friendly message) in case chosen becomes empty later:
    if not chosen:
        st.info("Select at least one manager to compute contributions.")
    else:
        rc_df = overall_return_contrib(panel[chosen], w)
        trc_df, port_var = overall_risk_contrib(panel[chosen], w)

        st.markdown("**Overall (annualised)**")
        st.dataframe(
            rc_df.join(trc_df[["TRC%"]], how="left").style.format(
                {"Mu(ann)": "{:.2%}", "RC": "{:.2%}", "RC%": "{:.1%}", "TRC%": "{:.1%}"}
            )
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
                pdf_bytes = build_pdf_full_report(
                    port=port,
                    panel_sel=panel[chosen],
                    chosen=chosen,
                    weights=w,
                    meta=meta,
                    template=template,
                    nbins=None,
                )
            except Exception:
                # Fallback to your existing builder if kaleido or pymupdf not available
                pdf_bytes = build_pdf(port, panel[chosen], w, meta)
            st.download_button("Save PDF", data=pdf_bytes, file_name="portfolio_report.pdf", mime="application/pdf")
    with c_share:
        st.button("Share / bookmark this setup", on_click=encode_state_to_query)
    st.caption("Click the button, then copy the URL from your browser.")
