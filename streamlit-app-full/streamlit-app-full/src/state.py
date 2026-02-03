from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import pandas as pd

def _get_query_params():
    try:
        return dict(st.query_params)
    except Exception:
        return dict(st.experimental_get_query_params())

def _set_query_params(d: dict):
    try:
        st.query_params.clear()
        st.query_params.update(d)
    except Exception:
        st.experimental_set_query_params(**d)

def load_state_from_query():
    qp = _get_query_params()
    if not qp:
        return
    if "m" in qp and qp["m"]:
        st.session_state["_chosen"] = qp["m"].split(",") if isinstance(qp["m"], str) else qp["m"]
    if "w" in qp and qp["w"]:
        try:
            d = json.loads(qp["w"])
            st.session_state["_weights"] = pd.Series(d, dtype=float)
        except Exception:
            pass
    if "t" in qp and qp["t"]:
        try:
            st.session_state["_start_ts"] = pd.to_datetime(qp["t"])
        except Exception:
            pass
    if "p" in qp and qp["p"]:
        try:
            st.session_state["_params"] = json.loads(qp["p"])
        except Exception:
            pass

def encode_state_to_query():
    qp = {}
    chosen = st.session_state.get("_chosen", [])
    if chosen:
        qp["m"] = ",".join(chosen)
    w = st.session_state.get("_weights")
    if w is not None and not w.empty:
        qp["w"] = json.dumps({k: float(v) for k, v in w.items()})
    ts = st.session_state.get("_start_ts")
    if ts is not None:
        qp["t"] = str(pd.Timestamp(ts).date())
    p = st.session_state.get("_params", {})
    if p:
        qp["p"] = json.dumps(p)
    _set_query_params(qp)

def apply_theme(dark: bool):
    st.session_state["_dark_mode"] = bool(dark)
    template = "plotly_dark" if dark else "plotly_white"
    st.session_state["_plotly_template"] = template
    if dark:
        st.markdown(
            """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        </style>
        """, unsafe_allow_html=True)

def get_plotly_template() -> str:
    return st.session_state.get("_plotly_template", "plotly_white")

def app_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd()
