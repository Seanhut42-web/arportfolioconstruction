from __future__ import annotations
from pathlib import Path
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from src.ingest import (
    FREQ_ME, read_manager_sheet, read_fx_sheet,
    infer_frequency, monthlyize_if_needed
)

def build_hedging_inputs(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheets = xls.sheet_names

    def try_fx(sh):
        try:
            return read_fx_sheet(xls, sh)
        except Exception:
            return None

    fx_series = try_fx(sheets[-1])
    fx_sheet = sheets[-1] if fx_series is not None else None
    if fx_series is None:
        for sh in sheets:
            fx_series = try_fx(sh)
            if fx_series is not None:
                fx_sheet = sh
                break
    if fx_series is None:
        raise ValueError("Could not identify FX sheet for hedging inputs.")

    fx_ret_m = fx_series.resample(FREQ_ME).last().pct_change().dropna()
    fx_ret_m.index = fx_ret_m.index + MonthEnd(0)

    man_local_m, man_ccy = {}, {}
    for sh in [s for s in sheets if s != fx_sheet]:
        df = read_manager_sheet(xls, sh)
        if df.empty or "Date" not in df or "Return" not in df:
            continue
        freq = infer_frequency(
            df["Date"],
            df["Frequency"].iloc[0] if "Frequency" in df.columns and not df["Frequency"].isna().all() else None
        )
        ccy = df["Currency"].dropna().iloc[0].upper() if "Currency" in df.columns and not df["Currency"].isna().all() else "GBP"
        s = pd.Series(df["Return"].values, index=pd.to_datetime(df["Date"]).tz_localize(None))
        s = s.sort_index()
        local_m = monthlyize_if_needed(s, freq)
        if not local_m.empty:
            man_local_m[sh] = local_m
            man_ccy[sh] = ccy

    if not man_local_m:
        raise ValueError("No valid manager series for hedging inputs.")
    all_months = sorted(set().union(*[s.index for s in man_local_m.values()]))
    span = (pd.Index(all_months).min(), pd.Index(all_months).max())
    return man_local_m, man_ccy, fx_ret_m, span


def apply_partial_hedge(local_ret_m: pd.Series, ccy: str, fx_ret_m: pd.Series, hedge_ratio: float = 1.0) -> pd.Series:
    local_ret_m = local_ret_m.sort_index()
    if ccy.upper() != "USD":
        return local_ret_m
    fx_aligned = fx_ret_m.reindex(local_ret_m.index).fillna(0.0)
    unhedged = (1.0 + local_ret_m) * (1.0 + fx_aligned) - 1.0
    fully_hedged = local_ret_m
    return hedge_ratio * fully_hedged + (1.0 - hedge_ratio) * unhedged


def build_panel_for_selection(man_local_m, man_ccy, fx_ret_m, chosen, mode, h_ratio, gbp_ann, usd_ann, start_ts):
    gbp_m = (1.0 + float(gbp_ann)) ** (1 / 12) - 1.0
    usd_m = (1.0 + float(usd_ann)) ** (1 / 12) - 1.0
    carry_m = (1.0 + gbp_m) / (1.0 + usd_m) - 1.0

    idx_union = sorted(set().union(*[man_local_m[m].loc[lambda s: s.index >= start_ts].index for m in chosen])) if chosen else []
    idx_union = pd.Index(idx_union)

    cols = {}
    for m in chosen:
        local = man_local_m[m].copy()
        local = local.loc[local.index >= start_ts]
        if local.empty:
            cols[m] = pd.Series(index=idx_union, dtype=float)
            continue
        if man_ccy.get(m, "GBP") == "USD":
            fx_m = fx_ret_m.reindex(local.index).fillna(0.0)
            unhedged = (1.0 + local) * (1.0 + fx_m) - 1.0
            hedged   = (1.0 + local) * (1.0 + carry_m) - 1.0
            series = unhedged if mode == "spot" else h_ratio * hedged + (1.0 - h_ratio) * unhedged
        else:
            series = local
        cols[m] = series.reindex(idx_union)

    panel = pd.DataFrame(cols, index=idx_union)
    panel.index.name = "Month"
    return panel
