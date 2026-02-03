from __future__ import annotations
import numpy as np
import pandas as pd

def overall_return_contrib(panel: pd.DataFrame, w: pd.Series, ann: int = 12) -> pd.DataFrame:
    panel = panel.copy()
    w = w.reindex(panel.columns).fillna(0.0)
    mu_ann = panel.mean() * ann
    port_ret_ann = float((w * mu_ann).sum())
    rc = w * mu_ann
    rc_share = rc / port_ret_ann if port_ret_ann != 0 else rc * 0
    out = pd.DataFrame({"Mu(ann)": mu_ann, "RC": rc, "RC%": rc_share})
    return out

def overall_risk_contrib(panel: pd.DataFrame, w: pd.Series, ann: int = 12):
    panel = panel.copy()
    w = w.reindex(panel.columns).fillna(0.0)
    Sigma = panel.cov() * ann
    port_var = float(w @ Sigma @ w)
    if Sigma.isna().any().any() or port_var <= 0:
        mrc = pd.Series(index=panel.columns, dtype=float)
        trc = pd.Series(index=panel.columns, dtype=float)
        share = pd.Series(index=panel.columns, dtype=float)
    else:
        mrc = Sigma @ w
        trc = w * mrc
        share = trc / port_var
    out = pd.DataFrame({"MRC": mrc, "TRC": trc, "TRC%": share})
    return out, port_var

def rolling_contrib(panel: pd.DataFrame, w: pd.Series, window: int = 36, ann: int = 12) -> pd.DataFrame:
    panel = panel.copy()
    w = w.reindex(panel.columns).fillna(0.0)
    if len(panel) < window:
        return pd.DataFrame()
    rows = []
    idx = []
    for end in range(window, len(panel) + 1):
        df = panel.iloc[end - window : end]
        mu_ann = df.mean() * ann
        Sigma = df.cov() * ann
        port_var = float(w @ Sigma @ w) if not Sigma.isna().any().any() else np.nan
        rc = w * mu_ann
        rc_share = rc / rc.sum() if rc.sum() != 0 else rc * 0
        if port_var == port_var and port_var > 0:
            mrc = Sigma @ w
            trc = w * mrc
            trc_share = trc / port_var
        else:
            trc_share = pd.Series(0.0, index=df.columns)
        frame = pd.DataFrame({"RC%": rc_share, "TRC%": trc_share}).T
        rows.append(frame)
        idx.append(df.index[-1])
    out = pd.concat(rows, keys=idx)
    return out
