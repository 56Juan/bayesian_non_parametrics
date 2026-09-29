r"""
diagnostico_gating.py — ¿Esta "despierto" el gating del PSBP? (experimento [EXP-1])
===================================================================================

Mide, sobre las trazas de UNA (componente FPCA, cadena), cuanto dependen del
estado los pesos del stick-breaking probit de `psbp_train.m`,

    v_h(x) = Phi(eta_h(x)),   eta_h(x) = alpha_h - sum_j psi_hj |x_j - Gamma_hj|,

con cantidades INVARIANTES a la permutacion de etiquetas de la mezcla:

    TV          promedio sobre los origenes de train de la variacion total
                entre pi(x_i) y su promedio: 0 = pesos constantes (gating
                dormido), 1 = pesos que cambian por completo con el estado.
    sd_eta      dispersion a traves de i del argumento probit, ponderada por la
                masa de cada quiebre. La calibracion de 101-109 apunta a 0.8
                (`sd_gate_objetivo`); por debajo de ~0.3 el gating es decorativo.
    gamma_on    fraccion (ponderada por masa) de pares (quiebre, covariable) con
                gamma = 1.
    psi_rel     psi posterior / mupsij donde gamma = 1: cuanto encoge la
                posterior al gating respecto del prior.
    n_atomos    atomos con masa > 5 %.

Es la misma cuenta del analisis que motivo [EXP-1]; se deja como funcion para
que el notebook la aplique igual a las trazas de la 107 y a las del experimento.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["pesos_por_draw", "diagnostico_gating"]


def pesos_por_draw(traces: dict, burn: int, X: np.ndarray, cada: int = 10):
    """eta (S, N-1, n) y pi (S, N, n) sobre las extracciones post-burn, 1 de cada `cada`."""
    sl = slice(int(burn), None, int(cada))
    A = np.asarray(traces["alphahout"], float)[sl]
    G = np.asarray(traces["Gammajhout"], float)[sl]
    P = np.asarray(traces["psijhout"], float)[sl]
    if G.ndim == 2:                       # p = 1: MATLAB colapsa el eje final
        G, P = G[..., None], P[..., None]
    eta = A[:, :, None] - np.einsum("shj,shij->shi", P, np.abs(X[None, None] - G[:, :, None, :]))
    v = norm.cdf(eta)
    cp = np.cumprod(1.0 - v, axis=1)
    prev = np.concatenate([np.ones_like(v[:, :1]), cp[:, :-1]], axis=1)
    pi = np.concatenate([v * prev, cp[:, -1:]], axis=1)
    return eta, pi, sl


def diagnostico_gating(traces: dict, burn: int, X: np.ndarray, mupsij, cada: int = 10) -> dict:
    """Las cinco cifras del docstring del modulo para una cadena."""
    X = np.asarray(X, float)
    eta, pi, sl = pesos_por_draw(traces, burn, X, cada)
    P = np.asarray(traces["psijhout"], float)[sl]
    g = np.asarray(traces["gammajhout"], float)[sl]
    if P.ndim == 2:
        P, g = P[..., None], g[..., None]
    g = g[:, : P.shape[1]]                                   # quiebres 1..N-1
    mup = np.atleast_1d(np.asarray(mupsij, float))

    tv = 0.5 * np.abs(pi - pi.mean(axis=2, keepdims=True)).sum(axis=1).mean(axis=1)
    masa = pi[:, :-1].mean(axis=2)                           # (S, N-1)
    sd_eta = (eta.std(axis=2) * masa).sum(axis=1) / masa.sum(axis=1)
    gamma_on = (g * masa[..., None]).sum(axis=(1, 2)) / (masa.sum(axis=1) * X.shape[1])
    with np.errstate(invalid="ignore"):
        psi_rel = np.nanmean(np.where(g == 1, P / mup[None, None, :], np.nan))
    n_atomos = (pi.mean(axis=2) > 0.05).sum(axis=1)
    return {"TV": float(np.median(tv)), "sd_eta": float(np.median(sd_eta)),
            "gamma_on": float(np.mean(gamma_on)), "psi_rel": float(psi_rel),
            "n_atomos": float(np.median(n_atomos))}
