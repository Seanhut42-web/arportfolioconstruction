
from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd

FREQ_ME = 'M'  # canonical monthly frequency token


def _compound_to_period(group: pd.Series) -> float:
    return (1 + group.fillna(0)).prod() - 1


def infer_frequency(idx: pd.DatetimeIndex) -> str:
    if len(idx) < 3:
        return 'M'
    deltas = np.diff(idx.values).astype('timedelta64[D]').astype(int)
    med = np.median(deltas)
    if med <= 2:
        return 'D'
    if med <= 10:
        return 'W'
    if med <= 40:
        return 'M'
    if med <= 120:
        return 'Q'
    return 'M'


def monthlyize_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    freq = infer_frequency(df.index)
    if freq == 'M':
        return df
    # compound to month-end
    return df.resample('M').apply(_compound_to_period).dropna(how='all')


def read_manager_sheet(file_or_path: io.BytesIO | str | Path | None = None) -> pd.DataFrame:
    """Read manager returns. CSV expected with Date column and manager columns in decimal returns.
    Falls back to demo file if not provided.
    """
    if file_or_path is None:
        demo = Path(__file__).resolve().parents[1] / 'data' / 'demo' / 'demo_managers.csv'
        return _read_returns_csv(demo)
    return _read_returns_csv(file_or_path)


def _read_returns_csv(p: io.BytesIO | str | Path) -> pd.DataFrame:
    if isinstance(p, (str, Path)):
        p = Path(p)
        if p.suffix.lower() == '.xlsx':
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)
    else:
        # bytes-like: try csv then excel
        try:
            df = pd.read_csv(p)
        except Exception:
            p.seek(0)
            df = pd.read_excel(p)

    df.columns = [str(c).strip() for c in df.columns]
    dcol = next((c for c in df.columns if c.lower() in ('date','asof','as_of','timestamp')), df.columns[0])
    df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
    df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # percent->decimal heuristic
    if df.abs().max().max() > 2.0:
        df = df / 100.0
    df = df.dropna(how='all')
    df = monthlyize_if_needed(df)
    return df


def read_fx_sheet(file_or_path: io.BytesIO | str | Path | None = None) -> pd.DataFrame:
    """Optional FX series; return empty if not provided."""
    if file_or_path is None:
        return pd.DataFrame()
    try:
        return _read_returns_csv(file_or_path)
    except Exception:
        return pd.DataFrame()
