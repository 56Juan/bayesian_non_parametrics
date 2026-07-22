"""
quadrature.py
=============
Cuadratura numerica sobre la grilla del dominio funcional.

Estas rutinas son la unica definicion de la regla de integracion empleada en
todo el proyecto. Cualquier cantidad que involucre una integral sobre el
dominio ---el producto interno de L^2, la matriz de Gram de una base, la norma
de un operador integral, el MISE--- debe construirse a partir de aqui: si dos
partes del pipeline usaran reglas distintas, la ortonormalidad de las
autofunciones FPCA dejaria de verificarse y el error seria silencioso.

Se emplea la regla trapezoidal, coherente con la grilla que incluye ambos
extremos del dominio (tau_1 = 0, tau_L = 1) definida en el esquema de
observacion de los algoritmos de simulacion.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pesos_trapezoidales",
    "integrar",
    "producto_interno",
    "gram",
    "norma_L2",
]


def pesos_trapezoidales(tau: np.ndarray) -> np.ndarray:
    """
    Pesos de la cuadratura trapezoidal asociada a la grilla `tau`:

        int f(s) ds ~= sum_l w_l f(tau_l).

    Para una grilla equiespaciada de paso h se obtiene w_1 = w_L = h/2 y
    w_l = h en los puntos interiores. La formula por diferencias centradas
    empleada aqui admite tambien grillas no equiespaciadas. Los pesos suman
    la longitud del dominio.
    """
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or tau.size < 2:
        raise ValueError(f"tau debe ser 1D con al menos 2 puntos; recibido {tau.shape}.")
    w = np.empty_like(tau)
    w[1:-1] = (tau[2:] - tau[:-2]) / 2.0
    w[0] = (tau[1] - tau[0]) / 2.0
    w[-1] = (tau[-1] - tau[-2]) / 2.0
    return w


def integrar(F: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Integra sobre el dominio la ultima dimension de `F`.

    F : (..., L) valores evaluados en la grilla.
    Retorna un arreglo de forma (...) con la integral de cada fila.
    """
    w = pesos_trapezoidales(tau)
    F = np.asarray(F, dtype=float)
    if F.shape[-1] != w.size:
        raise ValueError(
            f"La ultima dimension de F ({F.shape[-1]}) no coincide con la "
            f"grilla ({w.size})."
        )
    return F @ w


def producto_interno(F: np.ndarray, G: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Producto interno de L^2 entre filas de F y G, evaluadas en la grilla."""
    return integrar(np.asarray(F, dtype=float) * np.asarray(G, dtype=float), tau)


def gram(Phi: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Matriz de Gram de un sistema de funciones base evaluado en la grilla.

    Phi : (G, K) columna k = phi_k(tau_l).
    Retorna W (K, K) con W[j, k] = <phi_j, phi_k>_{L^2}, simetrizada para
    eliminar la asimetria numerica residual.
    """
    Phi = np.asarray(Phi, dtype=float)
    if Phi.ndim != 2:
        raise ValueError(f"Phi debe ser 2D (G, K); recibido {Phi.shape}.")
    w = pesos_trapezoidales(tau)
    if Phi.shape[0] != w.size:
        raise ValueError(
            f"Phi tiene {Phi.shape[0]} filas y la grilla {w.size} puntos."
        )
    W = Phi.T @ (w[:, None] * Phi)
    return 0.5 * (W + W.T)


def norma_L2(F: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Norma de L^2 de cada fila de F (..., L)."""
    return np.sqrt(integrar(np.asarray(F, dtype=float) ** 2, tau))
