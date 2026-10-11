"""
resumenes_predictiva.py
=======================
Predictores puntuales de la curva a partir de la predictiva del PSBPM-FD
(docs/03 Modelo.tex §03_05_04). Una sola definicion de cada funcional, sobre
extracciones de curvas `X` (S, n, G) o de scores (S, n, M):

    esperanza   la media ANALITICA de `momentos()`; no vive aqui. Nunca la media
                de las extracciones: con atau <= 1 la predictiva no tiene segundo
                momento y unos pocos draws la arrastran.
    mediana     mediana puntual, tau -> F^{-1}(1/2). Minimiza el MAE funcional
                punto a punto; puede ser un collage de tramos de curvas distintas.
    medoide     mediana de Frechet muestral en L^1: la extraccion que minimiza la
                suma de distancias L^1 a las demas. Es una curva que el modelo genera.
    modal       atomo mas probable: por iteracion y componente la media del atomo
                de mayor peso, mediana sobre iteraciones y cadenas, transportada
                a la curva (`atomo_modal`).

El medoide es O(S log S) por (origen, tau): la suma de distancias L^1 se
descompone por punto del dominio y, ordenando los S valores, sale de sumas
acumuladas. No se forma ninguna matriz S x S. La mediana por profundidad de banda
(MBD) estuvo aqui y se elimino: el estudio reporta los cuatro predictores de docs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .metrics_puntual import pesos_normalizados

__all__ = [
    "PREDICTORES", "mediana_puntual", "suma_desvios_absolutos", "medoide_l1",
    "atomo_modal", "distancia_a_muestra_mas_cercana",
]

#: Los cuatro predictores del estudio, en el orden en que se reportan.
PREDICTORES = ("esperanza", "mediana", "medoide", "modal")

_TROZO = 64     # origenes por trozo: acota la memoria de los temporales (S, trozo, G)


def _trozos(n: int):
    for i0 in range(0, n, _TROZO):
        yield slice(i0, min(n, i0 + _TROZO))


def mediana_puntual(X: np.ndarray) -> np.ndarray:
    """(S, n, G) -> (n, G): mediana de las extracciones en cada punto."""
    return np.median(np.asarray(X), axis=0)


def suma_desvios_absolutos(X: np.ndarray) -> np.ndarray:
    """
    (S, ...) -> (S, ...): sum_r |X_s - X_r| a lo largo del eje 0, por
    ordenamiento. Para el valor de rango i en la muestra ordenada `a`,
    sum_r |a_i - a_r| = a_i i - A_{i-1} + (A_S - A_i) - a_i (S - 1 - i),
    con A las sumas acumuladas.
    """
    X = np.asarray(X, dtype=float)
    S = X.shape[0]
    orden = np.argsort(X, axis=0)
    a = np.take_along_axis(X, orden, axis=0)
    A = np.cumsum(a, axis=0)
    i = np.arange(S, dtype=float).reshape((S,) + (1,) * (X.ndim - 1))
    sad = a * i - (A - a) + (A[-1:] - A) - a * (S - 1 - i)
    out = np.empty_like(sad)
    np.put_along_axis(out, orden, sad, axis=0)
    return out


def medoide_l1(X: np.ndarray, tau: np.ndarray, pesos_tau=None,
               devolver_indice: bool = False):
    """
    (S, n, G) -> (n, G): la extraccion que minimiza sum_r int |X_s - X_r| dtau,
    con la cuadratura comun (`pesos_normalizados`; la normalizacion no cambia el
    argmin). Con `devolver_indice` retorna tambien s* (n,).
    """
    X = np.asarray(X)
    S, n, G = X.shape
    w = pesos_normalizados(tau, pesos_tau)
    idx = np.empty(n, dtype=int)
    for sl in _trozos(n):
        D = suma_desvios_absolutos(X[:, sl]) @ w                  # (S, trozo)
        idx[sl] = np.argmin(D, axis=0)
    out = X[idx, np.arange(n)]
    return (out, idx) if devolver_indice else out


def atomo_modal(models_chains: Dict[int, Dict[int, object]], dfs: Dict[int, object]) -> np.ndarray:
    """
    (n, M): score modal por componente (docs eq. atomo_modal). Por iteracion y
    componente, la media del atomo de mayor peso (`ModeloTraza.atomo_modal`);
    MEDIANA sobre iteraciones y cadenas juntas, no promedio: las iteraciones en
    que dos atomos se reparten el peso no desplazan el predictor hacia el valle.
    `models_chains[k][c]` como en `cargar_trazas`; `dfs[k]` el dataset de k.
    """
    cols = []
    for k in sorted(models_chains):
        v = np.vstack([models_chains[k][c].atomo_modal(dfs[k]) for c in sorted(models_chains[k])])
        cols.append(np.median(v, axis=0))
    return np.column_stack(cols)


def distancia_a_muestra_mas_cercana(X: np.ndarray, curva: np.ndarray, tau: np.ndarray,
                                    pesos_tau=None, n_ref: int = 100,
                                    seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Control del collage de la mediana puntual (docs 03_05_04_01): distancia L^1
    de `curva` (n, G) a la extraccion mas cercana de `X` (S, n, G), contra la
    distancia TIPICA de una extraccion a su vecina mas cercana (mediana sobre
    `n_ref` extracciones de referencia, cada una contra las otras S - 1).
    Retorna `d_curva`, `d_tipica` y `razon = d_curva / d_tipica`, todas (n,).
    Una razon cercana a 1 dice que la curva es como una extraccion; muy por
    encima, que no es una curva que el modelo genere.
    """
    X = np.asarray(X)
    S, n, G = X.shape
    w = pesos_normalizados(tau, pesos_tau)
    rng = np.random.default_rng(seed)
    ref = rng.choice(S, size=min(int(n_ref), S), replace=False)
    d_curva, d_tip = np.empty(n), np.empty(n)
    for i in range(n):
        Xi = np.asarray(X[:, i], dtype=float)
        d_curva[i] = float((np.abs(Xi - curva[i][None, :]) @ w).min())
        D = np.abs(Xi[ref][:, None, :] - Xi[None, :, :]) @ w     # (n_ref, S)
        D[np.arange(len(ref)), ref] = np.inf
        d_tip[i] = float(np.median(D.min(axis=1)))
    return {"d_curva": d_curva, "d_tipica": d_tip,
            "razon": d_curva / np.maximum(d_tip, 1e-300)}
