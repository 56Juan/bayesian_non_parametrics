"""
diagnostics.py
--------------
Diagnósticos de convergencia MCMC para el modelo PSBP-FD.
"""
from __future__ import annotations
import numpy as np


def gelman_rubin(chains: list[np.ndarray]) -> float:
    """
    R-hat de Gelman-Rubin para múltiples cadenas.

    Parameters
    ----------
    chains : lista de arrays (nsim,) — una por cadena

    Returns
    -------
    R_hat : escalar (< 1.1 indica convergencia)
    """
    m  = len(chains)
    n  = chains[0].shape[0]
    chain_means = np.array([c.mean() for c in chains])
    grand_mean  = chain_means.mean()
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    W = np.mean([c.var(ddof=1) for c in chains])
    var_hat = (1 - 1/n) * W + B / n
    return float(np.sqrt(var_hat / W))


def effective_sample_size(chain: np.ndarray) -> float:
    """ESS usando la suma de autocorrelaciones."""
    n    = len(chain)
    c    = chain - chain.mean()
    acf  = np.correlate(c, c, mode="full")[n - 1:]
    acf /= acf[0]
    # Suma hasta el primer par negativo (regla de Geyer)
    rho_sum = 1.0
    for t in range(1, n // 2):
        pair = acf[2*t - 1] + acf[2*t]
        if pair < 0:
            break
        rho_sum += 2 * acf[t]
    return float(n / rho_sum)
