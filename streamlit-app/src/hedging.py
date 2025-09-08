
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from src.ingest import (
    FREQ_ME, read_manager_sheet, read_fx_sheet,
    infer_frequency, monthlyize_if_needed
)


def build_hedging_inputs() -> Dict:
    """Return default hedging controls (placeholder)."""
    return {
        'apply_fx_hedge': False,
        'hedge_weight': 0.0,
    }


def build_panel_for_selection(*args, **kwargs):
    """Compatibility shim — selection UI handled in pages."""
    return None


def apply_partial_hedge(series: pd.Series, hedge_weight: float = 0.0) -> pd.Series:
    """Toy partial hedge that reduces variance linearly; placeholder for real logic."""
    hedge_weight = max(0.0, min(1.0, hedge_weight))
    return series * (1.0 - hedge_weight)
