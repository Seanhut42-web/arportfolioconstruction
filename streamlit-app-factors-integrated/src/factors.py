from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.ingest import infer_frequency, to_monthly_compounded

FACTOR_DISPLAY = [
    "S&P500", "Global Credit", "Value", "Growth", "Momentum", "Size", "Quality", "Carry"
]

# ----- Load factors -----
def read_factors_excel(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Generic reader: expects first row with headers incl. 'Date'. Returns tidy DataFrame [Date, factor...]"""
    xls = pd.ExcelFile(path, engine="openpyxl")
    sh = sheet_name or xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sh)
    # Find date column heuristically
    date_col = None
    for c in df.columns:
        if str(c).strip().lower().startswith('date'):
            date_col = c; break
    if date_col is None:
        raise ValueError("Factors file must include a 'Date' column.")
    df = df.rename(columns={date_col: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['Date']).sort_values('Date')
    fac_cols = [c for c in df.columns if c != 'Date']
    for c in fac_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[['Date'] + fac_cols].dropna(how='all')


def read_factors_custom_layout(path: Path) -> pd.DataFrame:
    """Parse the user's 'Factor Returns.xlsx' layout:
    - Dates in column A from row 7 (1-based)
    - Prices from row 7 in columns B..I (8 factors)
    - Index labels in row 4 (B..I), but we override with canonical names in FACTOR_DISPLAY order.
    Returns a DataFrame with columns: Date + 8 factor price columns.
    """
    df = pd.read_excel(path, sheet_name=0, header=None, engine='openpyxl')
    # 0-based indexing: row 6 and below contain data
    data = df.iloc[6:, :]
    data = data.rename(columns={0: 'Date'})
    # Keep first 9 cols (A..I)
    data = data.iloc[:, :9]
    # Drop rows where Date is NaN
    data = data.dropna(subset=['Date'])
    # Convert Excel serial or date strings to datetime
    def to_dt(x):
        try:
            # If already datetime-like
            return pd.to_datetime(x).tz_localize(None)
        except Exception:
            try:
                # Excel serial numbers
                base = pd.Timestamp('1899-12-30')
                return base + pd.to_timedelta(int(float(x)), unit='D')
            except Exception:
                return pd.NaT
    data['Date'] = data['Date'].apply(to_dt)
    data = data.dropna(subset=['Date']).sort_values('Date')

    # Map factor price columns
    price_cols = {}
    for i, name in enumerate(FACTOR_DISPLAY, start=1):
        price_cols[i] = name
    data = data.rename(columns=price_cols)
    # Keep only the 8 factor columns we mapped
    cols = ['Date'] + FACTOR_DISPLAY
    data = data[cols]
    # Ensure numeric
    for c in FACTOR_DISPLAY:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna(how='all', subset=FACTOR_DISPLAY)
    return data


def to_monthly_returns_from_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    """From a DataFrame: index=Date, columns=Factor price levels → monthly returns in decimals.
    Uses month-end last price then pct_change.
    """
    df = df_prices.copy()
    df = df.set_index('Date')
    # Month-end last observation
    df_m = df.groupby(df.index.to_period('M')).last()
    df_m.index = df_m.index.to_timestamp('M')
    # Returns
    ret = df_m.pct_change().dropna(how='all')
    ret.index.name = 'Month'
    return ret


def coerce_to_monthly_returns(df: pd.DataFrame, assume_percent: bool | None = None) -> pd.DataFrame:
    """Compatibility shim: if fed generic data with returns, monthlyise/convert; here we mostly use custom layout."""
    # Detect if looks like prices (monotonic levels) vs returns
    fac_cols = [c for c in df.columns if c != 'Date']
    looks_like_price = False
    if fac_cols:
        s = pd.to_numeric(df[fac_cols[0]], errors='coerce')
        looks_like_price = s.dropna().abs().median() > 1.5  # crude heuristic
    if looks_like_price:
        return to_monthly_returns_from_prices(df[['Date'] + fac_cols])

    # Otherwise treat as returns; infer freq and monthlyise if needed
    freq = infer_frequency(df['Date'], None)
    df = df.set_index('Date')
    out = {}
    for c in fac_cols:
        s = pd.to_numeric(df[c], errors='coerce').dropna()
        if s.empty: continue
        if assume_percent:
            s = s / 100.0
        if freq in ("daily", "weekly"):
            out[c] = to_monthly_compounded(s)
        else:
            m = s.groupby(s.index.to_period('M')).last()
            m.index = m.index.to_timestamp('M')
            out[c] = m
    fac = pd.DataFrame(out)
    fac.index.name = 'Month'
    return fac

# ----- Regression helpers -----
def newey_west_ols(y: pd.Series, X: pd.DataFrame, add_const: bool = True, maxlags: int = 3):
    y = y.dropna()
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty:
        return None
    y2 = df.iloc[:, 0]
    X2 = df.iloc[:, 1:]
    if add_const:
        X2 = sm.add_constant(X2)
    model = sm.OLS(y2, X2).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return model


def compute_vif(X: pd.DataFrame) -> pd.Series:
    X = X.dropna()
    X = sm.add_constant(X)
    cols = X.columns
    vifs = []
    for i in range(1, len(cols)):
        vifs.append(variance_inflation_factor(X.values, i))
    return pd.Series(vifs, index=cols[1:], name='VIF')


def rolling_betas(y: pd.Series, X: pd.DataFrame, window: int = 36, add_const: bool = True) -> pd.DataFrame:
    df = pd.concat([y, X], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()
    y2 = df.iloc[:, 0]
    X2 = df.iloc[:, 1:]
    if add_const:
        X2 = sm.add_constant(X2)
    betas, idx = [], []
    for i in range(window, len(df) + 1):
        yi = y2.iloc[i - window:i]
        Xi = X2.iloc[i - window:i]
        try:
            res = sm.OLS(yi, Xi).fit()
            b = res.params.drop('const', errors='ignore')
            betas.append(b)
            idx.append(df.index[i - 1])
        except Exception:
            continue
    if not betas:
        return pd.DataFrame()
    return pd.DataFrame(betas, index=idx)

# ----- Helper: manager series per hedging settings -----
def series_gbp_for_manager(name: str, start_ts: pd.Timestamp, params: dict,
                           man_local_m: dict, man_ccy: dict, fx_ret_m: pd.Series) -> pd.Series:
    local = man_local_m[name]
    local = local.loc[local.index >= start_ts]
    if man_ccy.get(name, 'GBP') != 'USD':
        return local
    gbp_m = (1.0 + float(params.get('gbp_cash_ann', 0.05))) ** (1/12) - 1.0
    usd_m = (1.0 + float(params.get('usd_cash_ann', 0.05))) ** (1/12) - 1.0
    carry_m = (1.0 + gbp_m) / (1.0 + usd_m) - 1.0
    fx_m = fx_ret_m.reindex(local.index).fillna(0.0)
    unhedged = (1.0 + local) * (1.0 + fx_m) - 1.0
    hedged   = (1.0 + local) * (1.0 + carry_m) - 1.0
    if str(params.get('fx_mode','')).startswith('Unhedged'):
        return unhedged
    h = float(params.get('hedge_ratio', 1.0))
    return h * hedged + (1.0 - h) * unhedged

# ----- Convenience loader for the integrated file -----
def load_integrated_factors(data_dir: Path) -> pd.DataFrame:
    """Loads Factor Returns.xlsx from data/, converts to monthly returns with canonical names."""
    # Prefer the exact integrated filename
    p1 = data_dir / 'Factor Returns.xlsx'
    p2 = data_dir / 'factors.xlsx'
    if p1.exists():
        prices = read_factors_custom_layout(p1)
        fac = to_monthly_returns_from_prices(prices)
        fac.columns = FACTOR_DISPLAY
        return fac
    elif p2.exists():
        df = read_factors_excel(p2)
        return coerce_to_monthly_returns(df, assume_percent=None)
    else:
        raise FileNotFoundError('No integrated factors file found in data/. Expected "Factor Returns.xlsx" or "factors.xlsx".')
