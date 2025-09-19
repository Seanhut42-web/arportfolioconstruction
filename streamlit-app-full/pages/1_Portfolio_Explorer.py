import math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.hedging import build_hedging_inputs, build_panel_for_selection
from src.metrics import summarize, compute_drawdown
from src.contrib import overall_return_contrib, overall_risk_contrib, rolling_contrib
from src.report import build_pdf
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
        weights[m] = st.sidebar.slider(m, -1.0, 1.0, float(w0), 0.01)

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
            fig.update_layout(title="12‑month Rolling Volatility (Annualised)", hovermode="x unified",
                              xaxis_title="Date", yaxis_title="Volatility (ann.)",
                              margin=dict(l=40, r=20, t=60, b=40), template=template)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    elif chart == "Monthly Bars":
        dfb = port.to_frame("Monthly Return").reset_index(names="Month")
        fig_bar = px.bar(dfb, x="Month", y="Monthly Return", title="Portfolio Monthly Returns",
                         color="Monthly Return", color_continuous_scale="RdYlGn", template=template)
        fig_bar.update_yaxes(tickformat=".0%")
        fig_bar.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_bar, use_container_width=True)

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
        # Build Year×Month table and add YTD at the end
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
        fig_hm = px.imshow(piv.drop(columns=['YTD']), color_continuous_scale='RdYlGn', origin='upper', template=template)
            fig_hm.update_traces(
        # Overlay YTD column without background color
        for i, year in enumerate(piv.index):
            ytd_val = piv.loc[year, 'YTD']
            fig_hm.add_annotation(
                x='YTD', y=year,
                text=f'{ytd_val:.1%}' if pd.notna(ytd_val) else '',
                showarrow=False,
                font=dict(color='black'),
                align='center',
                bgcolor='white'
            )
                hovertemplate="Year=%{y}<br>Column=%{x}<br>Return=%{z:.1%}<extra></extra>",
                text=piv.applymap(lambda v: f"{v:.1%}" if pd.notna(v) else "").values,
                texttemplate="%{text}",
            )
            fig_hm.update_layout(title='Year × Month (Portfolio monthly returns) — with YTD',
        # Add black line to separate YTD column
        fig_hm.add_shape(
            type='line',
            x0=len(piv.columns) - 1.5, x1=len(piv.columns) - 1.5,
            y0=-0.5, y1=len(piv.index) - 0.5,
            line=dict(color='black', width=2)
        )
                                 xaxis_title='Month', yaxis_title='Year',
                                 coloraxis_colorbar=dict(title='Return', tickformat='.0%'),
                                 margin=dict(l=60, r=20, t=60, b=60))
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
        rc_ts  = roll.xs("RC%", level=1)
        st.area_chart(rc_ts)

    meta = st.session_state.get("_params", {})
    c_dl, c_share = st.columns([1,1])
    with c_dl:
        if st.button("Download PDF report", type="primary"):
            pdf_bytes = build_pdf(port, panel[chosen], w, meta)
            st.download_button("Save PDF", data=pdf_bytes, file_name="portfolio_report.pdf", mime="application/pdf")
    with c_share:
        st.button("Share / bookmark this setup", on_click=encode_state_to_query)
        st.caption("Click the button, then copy the URL from your browser.")
