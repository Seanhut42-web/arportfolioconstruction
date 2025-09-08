import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About this app")

st.markdown(
    """
This Streamlit app parses a multi‑sheet Excel workbook of manager track records, monthlyises mixed‑frequency data,
converts USD managers to GBP (unhedged or hedged via a CIP proxy), and lets you build custom portfolios.

- **Portfolio Explorer** (main page): select managers with checkboxes, set weights, choose start date, set hedging mode & ratio, and visualise results.
- **Manager Explorer**: single‑manager view with hedge ratio slider.

If you need enhancements (exports, costs, more CCYs), shout.
    """
)
