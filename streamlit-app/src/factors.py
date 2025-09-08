
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
except Exception:
    sm = None


def read_factor_data(file_or_df, date_col: str = 'Date', pct_to_decimal: bool = True) -> pd.DataFrame:
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        try:
            df = pd.read_csv(file_or_df)
        except Exception:
            file_or_df.seek(0)
            df = pd.read_excel(file_or_df)
    df.columns = [str(c).strip() for c in df.columns]
    if date_col not in df.columns:
        candidates = [c for c in df.columns if c.lower() in ('date','asof','as_of')]
        if not candidates:
            raise ValueError(f"Could not find date column '{date_col}'.")
        date_col = candidates[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if pct_to_decimal and df.abs().max().max() > 2.0:
        df = df / 100.0
    return df.dropna(how='all')


def align_series(r: pd.Series | pd.DataFrame, F: pd.DataFrame):
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    df = pd.concat([r.rename('ret'), F], axis=1, join='inner').dropna()
    return df['ret'], df.drop(columns=['ret'])


def run_factor_regression(r, F, add_const: bool = True, nw_lags: int | None = None):
    y, X = align_series(r, F)
    if sm is None:
        X_ = np.c_[np.ones(len(X)), X.values] if add_const else X.values
        beta = np.linalg.lstsq(X_, y.values, rcond=None)[0]
        intercept = float(beta[0]) if add_const else np.nan
        betas = pd.Series(beta[1:] if add_const else beta, index=X.columns, name='beta')
        resid = y.values - X_.dot(beta)
        r2 = 1 - np.var(resid, ddof=X_.shape[1]) / np.var(y.values, ddof=1)
        return {
            'betas': betas,
            'tstats': pd.Series(index=betas.index, dtype=float),
            'intercept': intercept,
            'intercept_t': np.nan,
            'r2': float(r2),
            'n': int(len(y)),
            'residuals': pd.Series(resid, index=y.index, name='resid'),
        }
    X1 = sm.add_constant(X) if add_const else X
    if nw_lags is not None and nw_lags > 0:
        model = sm.OLS(y, X1).fit(cov_type='HAC', cov_kwds={'maxlags': int(nw_lags)})
    else:
        model = sm.OLS(y, X1).fit()
    params, tvals = model.params, model.tvalues
    intercept = params.get('const', np.nan)
    intercept_t = tvals.get('const', np.nan)
    betas = params.drop('const', errors='ignore').rename('beta')
    tstats = tvals.drop('const', errors='ignore').rename('t')
    return {
        'betas': betas,
        'tstats': tstats,
        'intercept': float(intercept),
        'intercept_t': float(intercept_t),
        'r2': float(model.rsquared),
        'n': int(model.nobs),
        'residuals': model.resid.rename('resid'),
        'fitted': model.fittedvalues.rename('fitted'),
    }


def rolling_factor_regression(r: pd.Series, F: pd.DataFrame, window: int = 36, min_obs: int = 24,
                               add_const: bool = True, nw_lags: int | None = None) -> pd.DataFrame:
    y, X = align_series(r, F)
    rows, idx = [], []
    for end in range(window, len(y) + 1):
        yw = y.iloc[end - window:end]
        Xw = X.loc[yw.index]
        if len(yw) < min_obs:
            continue
        res = run_factor_regression(yw, Xw, add_const=add_const, nw_lags=nw_lags)
        rows.append(res['betas'])
        idx.append(yw.index[-1])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=idx).sort_index()
