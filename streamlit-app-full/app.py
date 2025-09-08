import streamlit as st

st.set_page_config(page_title="Manager Portfolio Analytics", layout="wide")

st.title("Manager Portfolio Analytics (GBP base)")

st.markdown(
    """
Use the **Portfolio Explorer** page on the left to:
- tick managers (checkboxes),
- set weights (supports shorts; can normalize to 100%),
- pick start date,
- choose **Unhedged (spot)** vs **Fully hedged (CIP proxy)** and a **hedge ratio**,
then press **Run portfolio analytics**. Results persist across tabs.

See **Manager Explorer** for single‑manager hedging comparisons.
    """
)
