"""
sim_escenario_C.py
==================
Base de Fourier ortonormal sobre [0, 1] que comparten los generadores sobre
scores (`sim_scores_comun`, `sim_escenario_TAR`, `sim_escenario_C1`).

Los coeficientes sobre esta base SON los scores: no hay FPCA que estimar ni base
que elegir por GCV, de modo que la discrepancia entre prediccion y objetivo
proviene de la dinamica y no de la representacion.

El generador de la familia C original (indice cuadratico sobre varios rezagos)
vivia aqui y se elimino: no lo usaba ninguna corrida.
"""

from __future__ import annotations

import numpy as np

from .sim_comun import pesos_trapezoidales

__all__ = ["base_fourier", "gram_base"]


def base_fourier(tau: np.ndarray, J: int) -> np.ndarray:
    """
    Sistema ortonormal de Fourier sobre [0, 1], SIN el termino constante:

        phi_{2k-1}(tau) = sqrt(2) sin(2 pi k tau)
        phi_{2k}(tau)   = sqrt(2) cos(2 pi k tau),      k = 1, 2, ...

    El constante se omite porque el nivel lo aporta mu; si estuviera, el
    coeficiente a_t1 y la media serian la misma direccion y lambda_1 dejaria de
    ser interpretable como la varianza de una componente.

    Retorna (L, J). J debe ser par para que el sistema quede cerrado por pares
    seno/coseno y el espectro no favorezca arbitrariamente una fase.
    """
    tau = np.asarray(tau, dtype=float)
    if J < 2 or J % 2 != 0:
        raise ValueError(f"J={J}: debe ser un entero par >= 2.")
    Phi = np.empty((tau.size, J))
    for k in range(1, J // 2 + 1):
        Phi[:, 2 * k - 2] = np.sqrt(2.0) * np.sin(2.0 * np.pi * k * tau)
        Phi[:, 2 * k - 1] = np.sqrt(2.0) * np.cos(2.0 * np.pi * k * tau)
    return Phi


def gram_base(Phi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Gram de la base en la cuadratura del estudio; debe dar la identidad."""
    w = pesos_trapezoidales(tau)
    return Phi.T @ (w[:, None] * Phi)
