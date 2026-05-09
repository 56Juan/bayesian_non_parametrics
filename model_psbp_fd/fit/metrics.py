"""
metrics.py
----------
Métricas de evaluación para el modelo PSBP-FD.
"""
from __future__ import annotations
import numpy as np


def rmse_per_component(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """RMSE por coeficiente funcional. Devuelve array (K,)."""
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


def lpml(log_lik_matrix: np.ndarray) -> float:
    """
    Log Pseudo-Marginal Likelihood (LPML) via CPO.

    Parameters
    ----------
    log_lik_matrix : (nsim_post, n) log-verosimilitudes por iteración y observación
    """
    # CPO_i = 1 / E[1/p(y_i | theta)]
    log_inv_mean = -np.log(np.mean(np.exp(-log_lik_matrix), axis=0))
    return float(log_inv_mean.sum())


def waic(log_lik_matrix: np.ndarray) -> float:
    """
    WAIC (Watanabe, 2010).
    log_lik_matrix : (nsim_post, n)
    """
    lppd    = np.log(np.mean(np.exp(log_lik_matrix), axis=0)).sum()
    p_waic2 = np.var(log_lik_matrix, axis=0, ddof=1).sum()
    return float(-2 * (lppd - p_waic2))
