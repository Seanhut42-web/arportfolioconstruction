from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def estimate_mu_cov(panel: pd.DataFrame, ann: int = 12):
    mu = panel.mean() * ann
    Sigma = panel.cov() * ann
    return mu, Sigma

def _bounds_vectors(n, bounds):
    lb, ub = (bounds[0] * np.ones(n)), (bounds[1] * np.ones(n))
    return lb, ub

def _sum_to_one_constraint():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

def optimise_mvo(mu: pd.Series, Sigma: pd.DataFrame, objective="max_sharpe",
                 target_ret=None, target_vol=None, bounds=(0.0, 1.0), risk_free=0.0):
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    lb, ub = _bounds_vectors(n, bounds)
    bnds = tuple(zip(lb, ub))
    cons = [_sum_to_one_constraint()]
    def stats(w):
        r = float(w @ mu.values)
        v = float(w @ Sigma.values @ w)
        return r, np.sqrt(max(v, 0.0))
    if objective == "max_sharpe":
        def neg_sharpe(w):
            r, s = stats(w)
            return - (r - risk_free) / (s + 1e-12)
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnds, constraints=cons)
    elif objective == "target_return":
        assert target_ret is not None, "target_ret required"
        cons2 = cons + [{"type": "eq", "fun": lambda w: w @ mu.values - float(target_ret)}]
        res = minimize(lambda w: w @ Sigma.values @ w, w0, method="SLSQP", bounds=bnds, constraints=cons2)
    elif objective == "target_vol":
        assert target_vol is not None, "target_vol required"
        def obj(w):
            r, s = stats(w)
            return (s - float(target_vol))**2 - 0.01 * r
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
    else:
        raise ValueError("Unknown objective")
    w = np.clip(res.x, lb, ub)
    w = w / w.sum() if w.sum() != 0 else w
    return pd.Series(w, index=mu.index)

def solve_risk_parity(Sigma: pd.DataFrame, bounds=(0.0, 1.0)):
    n = Sigma.shape[0]
    w0 = np.full(n, 1.0 / n)
    lb, ub = _bounds_vectors(n, bounds)
    bnds = tuple(zip(lb, ub))
    cons = [_sum_to_one_constraint()]
    def obj(w):
        mrc = Sigma.values @ w
        trc = w * mrc
        total_var = float(w @ Sigma.values @ w) + 1e-12
        c = trc / total_var
        return ((c - c.mean()) ** 2).sum()
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 1000})
    w = np.clip(res.x, lb, ub)
    w = w / w.sum() if w.sum() != 0 else w
    return pd.Series(w, index=Sigma.columns)
