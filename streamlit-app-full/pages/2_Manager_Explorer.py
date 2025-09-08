from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.hedging import build_hedging_inputs, apply_partial_hedge
from src.metrics import summarize

st.set_page_config(page_title="Manager Explorer", layout="wide")

@st.cache_data(show_spinner="Loading inputs…", ttl=None)
def load_inputs():
    xlsx_path = Path(__file__).parent.parent / "data" / "Manager Track Records v2.xlsx"
    return build_hedging_inputs(xlsx_path)

man_local_m, man_ccy, fx_ret_m, span = load_inputs()

manager = st.selectbox("Manager", sorted(man_local_m.keys()))
h = st.slider("Hedge ratio (GBP base for USD managers)", 0.0, 1.0, 1.0, 0.05)
local = man_local_m[manager]
ccy = man_ccy[manager]

gbp = apply_partial_hedge(local, ccy, fx_ret_m, hedge_ratio=h)
st.caption(f"Currency: **{ccy}** | Period: {gbp.index.min().date()} → {gbp.index.max().date()}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=gbp.index, y=(1+gbp).cumprod(), name="GBP (selected hedge)", line=dict(width=3)))
fig.update_layout(title="Cumulative Growth of £1", hovermode="x unified", yaxis_title="Value (£)" )
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Last 12 months (GBP, selected hedge)**")
st.dataframe(gbp.tail(12).to_frame("Monthly return").style.format("{:.2%}"))

st.markdown("**Summary (GBP, selected hedge)**")
st.dataframe(pd.DataFrame(summarize(gbp), index=[0]).T.rename(columns={0: "Value"}).style.format({
    "Value": "{:.2%}",
}))
