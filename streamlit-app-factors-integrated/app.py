import streamlit as st

st.set_page_config(page_title="Manager Portfolio Analytics", layout="wide")

st.title("Manager Portfolio Analytics (GBP base)")

st.markdown(
    """
- Use **Portfolio Explorer** to select managers, hedging and date, then press **Run portfolio analytics**.
- Use **Manager Explorer** for a single manager view.
- Use **Factor Analysis** to estimate betas vs your factor set (bundled, no upload needed).
    """
)
