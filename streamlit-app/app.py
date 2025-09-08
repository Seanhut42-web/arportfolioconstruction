from pathlib import Path
import streamlit as st
from src.ingest import parse_or_load_cached

st.set_page_config(page_title="Manager Portfolio Analytics", layout="wide")

@st.cache_data(show_spinner="Parsing workbook…", ttl=None)
def load_panel():
    data_path = Path(__file__).parent / "data" / "Manager Track Records v2.xlsx"
    cache_path = Path(__file__).parent / "data" / "returns_monthly.parquet"
    df = parse_or_load_cached(data_path, cache_parquet=cache_path)
    return df

st.title("Manager Portfolio Analytics (GBP base)")

returns_df = load_panel()
st.success(f"Loaded {returns_df.shape[1]} managers | {returns_df.index.min().date()} → {returns_df.index.max().date()}")
st.dataframe(returns_df.tail().style.format("{:.2%}"))

st.markdown(
    """
Use the **Portfolio Explorer** page to select managers, set weights (incl. shorting), 
pick a start date, and choose **Unhedged (spot)** or **Fully hedged (CIP proxy)** FX handling with a **hedge ratio**.
    """
)
