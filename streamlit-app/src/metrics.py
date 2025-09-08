import math
import numpy as np
import pandas as pd


def compute_drawdown(cum: pd.Series) -> pd.Series:
    peak = cum.cummax()
    return cum / peak - 1.0


def summarize(ret_m: pd.Series, periods: int = 12) -> dict:
    if ret_m.empty:
        return {}
    cg = (1 + ret_m).prod()
    yrs = len(ret_m) / periods
    ann_ret = cg**(1 / yrs) - 1 if yrs > 0 else np.nan
    ann_vol = ret_m.std(ddof=0) * math.sqrt(periods)
    sharpe = ann_ret / ann_vol if (ann_vol and ann_vol > 0) else np.nan
    max_dd = compute_drawdown((1 + ret_m).cumprod()).min()
    calmar = ann_ret / abs(max_dd) if (isinstance(max_dd, (float, np.floating)) and max_dd < 0) else np.nan
    return {
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }
