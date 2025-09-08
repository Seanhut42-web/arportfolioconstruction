
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_drawdown(r: pd.Series) -> pd.Series:
    cum = (1 + r.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd


def summarize(r: pd.Series | pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
    """Return basic performance stats.
    If DataFrame, compute per column.
    """
    if isinstance(r, pd.DataFrame):
        return pd.concat({c: summarize(r[c], freq).iloc[:, 0] for c in r.columns}, axis=1)

    r = r.dropna()
    if len(r) == 0:
        return pd.DataFrame()

    periods_per_year = {
        'D': 252, 'W': 52, 'M': 12, 'Q': 4
    }.get(freq.upper(), 12)

    total_return = (1 + r).prod() - 1
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    vol_a = r.std(ddof=1) * np.sqrt(periods_per_year)
    sr = r.mean() * periods_per_year / vol_a if vol_a != 0 else np.nan

    dd = compute_drawdown(r)
    maxdd = dd.min()

    out = pd.DataFrame({
        'CAGR': [cagr],
        'Vol': [vol_a],
        'Sharpe': [sr],
        'MaxDD': [maxdd],
        'N': [len(r)],
    }).T
    return out
