"""
metrics_puntual.py
==================
Metricas de error puntual del desempeno predictivo (Seccion 2.2.3.2).

La evaluacion se organiza en dos niveles. El primero opera sobre los
coeficientes de la representacion funcional y se reporta componente a
componente, dado que la escala de los coeficientes depende del sistema
empleado y sus varianzas pueden diferir en ordenes de magnitud, de modo que un
promedio sin normalizar queda dominado por las componentes de mayor varianza y
pierde capacidad diagnostica. El segundo opera sobre la curva reconstruida y
captura simultaneamente el error de prediccion en los coeficientes y el error
de aproximacion de la representacion.

Todas las funciones comparan predicciones contra observaciones sin asumir
forma alguna para el mecanismo que las genera, de modo que permiten contrastar
en igualdad de condiciones modelos de naturaleza distinta.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..utils.quadrature import integrar

__all__ = [
    "rmse",
    "mse_por_coeficiente",
    "rmse_por_coeficiente",
    "r2_por_columna",
    "razon_dispersion",
    "mise",
    "rmse_funcional",
    "resumen_puntual",
]


def _2d(a) -> np.ndarray:
    return np.atleast_2d(np.asarray(a, dtype=float))


def rmse(y_obs, y_pred) -> float:
    """Raiz del error cuadratico medio sobre todos los elementos."""
    y_obs, y_pred = np.asarray(y_obs, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_obs) ** 2)))


def mse_por_coeficiente(alpha_obs, alpha_pred) -> np.ndarray:
    """
    Error cuadratico medio por coeficiente (Seccion 2.2.3.2, primer nivel).

    alpha_obs, alpha_pred : (n, K)
    Retorna (K,).
    """
    A, P = _2d(alpha_obs), _2d(alpha_pred)
    if A.shape != P.shape:
        raise ValueError(f"Formas incompatibles: {A.shape} vs {P.shape}.")
    return np.mean((P - A) ** 2, axis=0)


def rmse_por_coeficiente(alpha_obs, alpha_pred) -> np.ndarray:
    """Raiz del MSE por coeficiente."""
    return np.sqrt(mse_por_coeficiente(alpha_obs, alpha_pred))


def r2_por_columna(y_obs, y_pred, centrar: bool = True) -> np.ndarray:
    """
    Coeficiente de determinacion por columna.

    Con `centrar=True` el denominador emplea la media de cada columna, de modo
    que la metrica evalua unicamente la capacidad de predecir la variacion
    temporal de esa columna. Con `centrar=False` emplea la media global, y el
    resultado incorpora ademas el acierto en los niveles relativos entre
    columnas.

    Esa distincion no es cosmetica. Cuando las columnas tienen medias muy
    distintas entre si, la version no centrada queda dominada por la
    reproduccion de los niveles ---que la media de la representacion entrega
    sin necesidad de modelo dinamico alguno--- y arroja valores muy superiores
    a los de la version centrada. La identidad que las relaciona es

        R2_no_centrado = 1 - sum_j (1 - R2_j) SS_j / (SS_intra + SS_entre),

    con SS_j la suma de cuadrados centrada de la columna j, SS_intra su suma y
    SS_entre la dispersion de las medias por columna. Para juzgar capacidad
    predictiva dinamica debe usarse la version centrada.
    """
    A, P = _2d(y_obs), _2d(y_pred)
    if A.shape != P.shape:
        raise ValueError(f"Formas incompatibles: {A.shape} vs {P.shape}.")
    ref = A.mean(axis=0) if centrar else A.mean()
    ss_res = np.sum((P - A) ** 2, axis=0)
    ss_tot = np.sum((A - ref) ** 2, axis=0)
    return np.where(ss_tot > 0, 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0), np.nan)


def razon_dispersion(y_obs, y_pred) -> np.ndarray:
    """
    Razon sd(y_pred) / sd(y_obs) por columna.

    Para una media condicional bien calibrada esta razon aproxima la
    correlacion entre observado y predicho, puesto que la varianza de una
    esperanza condicional es siempre menor que la de la variable original. Una
    razon marcadamente inferior a la correlacion indica sobre-encogimiento; una
    superior, una prediccion mas volatil que la propia serie.
    """
    A, P = _2d(y_obs), _2d(y_pred)
    sd_o = A.std(axis=0, ddof=0)
    return np.where(sd_o > 0, P.std(axis=0, ddof=0) / np.where(sd_o > 0, sd_o, 1.0), np.nan)


def mise(X_obs, X_pred, tau) -> float:
    """
    Error cuadratico medio integrado de prediccion (Seccion 2.2.3.2).

    X_obs, X_pred : (n, G) curvas evaluadas en la grilla.
    Se promedia sobre los origenes y se integra sobre el dominio mediante la
    cuadratura comun del proyecto.
    """
    O, P = _2d(X_obs), _2d(X_pred)
    if O.shape != P.shape:
        raise ValueError(f"Formas incompatibles: {O.shape} vs {P.shape}.")
    return float(integrar(np.mean((P - O) ** 2, axis=0), tau))


def rmse_funcional(X_obs, X_pred, tau) -> float:
    """Raiz del MISE; expresa el desajuste en las unidades de la curva."""
    return float(np.sqrt(mise(X_obs, X_pred, tau)))


def resumen_puntual(y_obs, y_pred, X_obs=None, X_pred=None, tau=None,
                    etiquetas: Optional[list] = None):
    """
    Tabla de metricas puntuales por componente y, si se entregan curvas, a
    nivel funcional.

    y_obs, y_pred : (n, M) coeficientes o scores observados y predichos.
    X_obs, X_pred : (n, G) curvas, opcionales.
    Retorna un DataFrame con una fila por componente y, cuando corresponde,
    dos columnas adicionales con el MISE y el RMSE funcional del conjunto.
    """
    import pandas as pd

    A, P = _2d(y_obs), _2d(y_pred)
    M = A.shape[1]
    idx = etiquetas if etiquetas is not None else [f"comp_{m+1}" for m in range(M)]

    tabla = pd.DataFrame({
        "RMSE": rmse_por_coeficiente(A, P),
        "R2": r2_por_columna(A, P, centrar=True),
        "corr": [float(np.corrcoef(A[:, m], P[:, m])[0, 1]) for m in range(M)],
        "sd_pred/sd_obs": razon_dispersion(A, P),
        "n": A.shape[0],
    }, index=idx)
    tabla.index.name = "componente"

    if X_obs is not None and X_pred is not None:
        if tau is None:
            raise ValueError("Se requiere `tau` para las metricas funcionales.")
        tabla.attrs["MISE"] = mise(X_obs, X_pred, tau)
        tabla.attrs["RMSE_funcional"] = rmse_funcional(X_obs, X_pred, tau)
    return tabla
