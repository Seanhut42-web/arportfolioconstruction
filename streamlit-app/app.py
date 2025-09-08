
import streamlit as st
st.set_page_config(page_title="AR Portfolio Construction", layout="wide")

st.title("AR Portfolio Construction")

st.markdown(
    """
    **Welcome.** Use the pages on the left:
    - **Portfolio Explorer** – build a portfolio from manager returns (heatmap intact)
    - **Manager Explorer** – inspect a single manager
    - **Factor Attribution** – run OLS/HAC regressions on your portfolio
    """
)
