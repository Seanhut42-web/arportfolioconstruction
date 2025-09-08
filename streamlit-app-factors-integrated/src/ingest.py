from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd

FREQ_ME = "ME"

# --- helpers ---
def _dedup_columns(cols):
    seen, out = {}, []
    for c in cols:
        k = str(c).strip()
        n = seen.get(k, -1) + 1
        seen[k] = n
        out.append(k if n == 0 else f"{k}.{n}")
    return out

def _find_header_row(df_raw: pd.DataFrame) -> int:
    for i in range(len(df_raw)):
        row_vals = df_raw.iloc[i].astype(str).str.strip().str.lower().values
        if any(v == "date" for v in row_vals):
            return i
    return 0

def infer_frequency(dates: pd.Series, stated: str | None) -> str:
    if isinstance(stated, str):
        s = stated.lower()
        if "daily" in s: return "daily"
        if "weekly" in s: return "weekly"
        if "month" in s:  return "monthly"
    dt = pd.to_datetime(dates, errors="coerce")
    diff = dt.diff().dt.days.dropna()
    med = diff.median() if not diff.empty else 30
    if med <= 2:  return "daily"
    if med <= 9:  return "weekly"
    return "monthly"

def read_manager_sheet(xls, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()
    hdr = _find_header_row(raw)
    header = _dedup_columns(raw.iloc[hdr].astype(str).str.strip().tolist())
    df = raw.iloc[hdr + 1 :].copy()
    df.columns = header

    candidates = [c for c in df.columns if "date" in c.lower()] or list(df.columns)
    date_col, best_n, parsed = None, -1, None
    for c in candidates:
        tp = pd.to_datetime(df[c], errors="coerce")
        n = tp.notna().sum()
        if n > best_n:
            best_n, date_col, parsed = n, c, tp
    if date_col is None:
        return pd.DataFrame()

    df = df.loc[parsed.notna()].copy()
    df["Date"] = parsed[parsed.notna()]

    ret_col = next((c for c in df.columns if c.lower() == "return" or c.lower().startswith("return")), None)
    if ret_col is None: ret_col = next((c for c in df.columns if "ret" in c.lower()), None)
    if ret_col is None:
        nums = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Date"]
        ret_col = nums[0] if nums else None
    if ret_col is None:
        return pd.DataFrame()

    cur_col = next((c for c in df.columns if c.lower().startswith("currency")), None)
    frq_col = next((c for c in df.columns if c.lower().startswith("frequency")), None)

    keep = ["Date", ret_col] + ([cur_col] if cur_col else []) + ([frq_col] if frq_col else [])
    out = df[keep].copy()
    rename = {ret_col: "Return"}
    if cur_col: rename[cur_col] = "Currency"
    if frq_col: rename[frq_col] = "Frequency"
    out = out.rename(columns=rename)

    out["Return"] = pd.to_numeric(out["Return"], errors="coerce")
    out = out.dropna(subset=["Date", "Return"]).sort_values("Date").reset_index(drop=True)
    if "Currency" in out:  out["Currency"]  = out["Currency"].astype(str).str.strip()
    if "Frequency" in out: out["Frequency"] = out["Frequency"].astype(str).str.strip()
    return out

def read_fx_sheet(xls, sheet: str) -> pd.Series:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ValueError("FX sheet is empty.")
    hdr = _find_header_row(raw)
    header = _dedup_columns(raw.iloc[hdr].astype(str).str.strip().tolist())
    df = raw.iloc[hdr + 1 :].copy()
    df.columns = header

    candidates = [c for c in df.columns if "date" in c.lower()] or list(df.columns)
    date_col, best_n, parsed = None, -1, None
    for c in candidates:
        tp = pd.to_datetime(df[c], errors="coerce")
        n = tp.notna().sum()
        if n > best_n:
            best_n, date_col, parsed = n, c, tp
    if date_col is None:
        raise ValueError("FX sheet: couldn't identify a Date column.")

    df = df.loc[parsed.notna()].copy()
    df["Date"] = parsed[parsed.notna()]

    fx_col, invert = None, False
    for c in [c for c in df.columns if c != "Date"]:
        name = c.lower().replace(" ", "")
        if "usdgbp" in name: fx_col, invert = c, False; break
        if "gbpusd" in name: fx_col, invert = c, True;  break
    if fx_col is None:
        counts = []
        for c in df.columns:
            if c == "Date" or "return" in c.lower():
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            counts.append((c, s.notna().sum()))
        if not counts:
            raise ValueError("FX sheet: couldn't find a numeric rate column.")
        fx_col = max(counts, key=lambda x: x[1])[0]

    rate = pd.to_numeric(df[fx_col], errors="coerce")
    fx = pd.Series(rate.values, index=pd.to_datetime(df["Date"]).dt.tz_localize(None)).dropna().sort_index()
    fx = fx[~fx.index.duplicated(keep="last")]
    if invert:
        fx = 1.0 / fx
    return fx

def to_monthly_compounded(s: pd.Series) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    m = (1.0 + s).resample(FREQ_ME).prod() - 1.0
    m.index = m.index + MonthEnd(0)
    return m[~m.index.duplicated(keep="last")].dropna()

def monthlyize_if_needed(s: pd.Series, freq: str) -> pd.Series:
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    if freq in ("daily", "weekly"):
        return to_monthly_compounded(s)
    m = s.groupby(s.index.to_period("M")).last()
    m.index = m.index.to_timestamp("M")
    return m[~m.index.duplicated(keep="last")].dropna()
