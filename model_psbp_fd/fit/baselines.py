"""
baselines.py
============
Lineas base de referencia sobre el bloque de prueba.

Un modelo flexible solo justifica su complejidad si supera a predictores que no
requieren estimacion alguna. Se implementan dos: la media incondicional del
bloque de entrenamiento y la persistencia, que copia la ultima observacion
disponible. Ambas se evaluan con las mismas metricas que el modelo propuesto y
en los mismos dos niveles ---coeficientes y curva reconstruida--- de modo que
la comparacion es directa.

La persistencia merece atencion particular en series de baja autocorrelacion:
copiar la observacion anterior duplica la varianza de la innovacion, de modo
que puede resultar peor que la media incondicional a nivel de coeficientes y
mejor a nivel de curva, o viceversa. Que ambos niveles discrepen no es un
error sino informacion sobre donde se concentra el error de cada predictor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .metrics_puntual import (
    mise, r2_por_columna, rmse_funcional, rmse_por_coeficiente,
)

__all__ = [
    "prediccion_media_incondicional",
    "prediccion_persistencia",
    "evaluar_baseline",
    "tabla_baselines",
]


def prediccion_media_incondicional(SCORES_STD: np.ndarray, T0: int) -> np.ndarray:
    """
    Prediccion de la media incondicional del bloque de entrenamiento.

    En escala estandarizada con los momentos de entrenamiento, esa media es
    cero por construccion, de modo que la prediccion es identicamente nula. El
    RMSE resultante es aproximadamente uno y constituye la referencia natural:
    todo modelo con RMSE mayor que uno es peor que no modelar nada.
    """
    S = np.atleast_2d(np.asarray(SCORES_STD, dtype=float))
    _validar_T0(S.shape[0], T0)
    return np.zeros((S.shape[0] - T0, S.shape[1]))


def prediccion_persistencia(SCORES_STD: np.ndarray, T0: int,
                            h: int = 1) -> np.ndarray:
    """
    Prediccion de persistencia a horizonte h: la observacion h pasos atras.

    Para h = 1 y origen t, la prediccion es el score observado en t - 1, que es
    informacion disponible en el momento de predecir; los primeros origenes del
    bloque de prueba emplean observaciones del final del bloque de
    entrenamiento, lo que constituye el condicionamiento en la historia y no
    fuga de informacion.
    """
    S = np.atleast_2d(np.asarray(SCORES_STD, dtype=float))
    T = S.shape[0]
    _validar_T0(T, T0)
    if h < 1 or T0 - h < 0:
        raise ValueError(f"h={h} incompatible con T0={T0}.")
    return S[T0 - h: T - h, :]


def _validar_T0(T: int, T0: int) -> None:
    if not (0 < T0 < T):
        raise ValueError(f"T0={T0} debe estar en (0, T) con T={T}.")


def evaluar_baseline(
    nombre: str,
    Y_std_pred: np.ndarray,
    SCORES_STD: np.ndarray,
    T0: int,
    estandarizador=None,
    fpca=None,
    X_obs: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
) -> dict:
    """
    Evalua una prediccion del bloque de prueba en los dos niveles.

    Y_std_pred    : (n_test, M) prediccion en escala estandarizada.
    SCORES_STD    : (T, M) scores estandarizados de la serie completa.
    estandarizador: objeto con `inverse_transform`, para volver a escala original.
    fpca          : objeto con `reconstruct`, para pasar de scores a curvas.
    X_obs         : (T, G) curvas observadas; se recorta al bloque de prueba.
    tau           : grilla, requerida para las metricas funcionales.

    Si no se entregan `estandarizador`, `fpca`, `X_obs` y `tau`, se reportan
    unicamente las metricas de coeficientes.
    """
    S = np.atleast_2d(np.asarray(SCORES_STD, dtype=float))
    _validar_T0(S.shape[0], T0)
    Y_obs = S[T0:, :]
    Y_pred = np.atleast_2d(np.asarray(Y_std_pred, dtype=float))
    if Y_pred.shape != Y_obs.shape:
        raise ValueError(
            f"La prediccion tiene forma {Y_pred.shape} y el bloque de prueba "
            f"{Y_obs.shape}.")

    rmse_m = rmse_por_coeficiente(Y_obs, Y_pred)
    r2_m = r2_por_columna(Y_obs, Y_pred, centrar=True)

    fila = {"modelo": nombre, "n_test": int(Y_obs.shape[0])}
    fila.update({f"RMSE_fpc{m+1}": float(rmse_m[m]) for m in range(rmse_m.size)})
    fila["RMSE_scores_prom"] = float(rmse_m.mean())
    fila["R2_scores_prom"] = float(np.nanmean(r2_m))

    completos = all(x is not None for x in (estandarizador, fpca, X_obs, tau))
    if completos:
        S_pred = estandarizador.inverse_transform(Y_pred)
        X_pred = fpca.reconstruct(S_pred)
        X_test = np.atleast_2d(np.asarray(X_obs, dtype=float))[T0:, :]
        fila["MISE"] = mise(X_test, X_pred, tau)
        fila["RMSE_funcional"] = rmse_funcional(X_test, X_pred, tau)
    return fila


def tabla_baselines(SCORES_STD: np.ndarray, T0: int, estandarizador=None,
                    fpca=None, X_obs=None, tau=None, h: int = 1):
    """
    Tabla comparativa de las lineas base sobre el bloque de prueba.

    Retorna un DataFrame indexado por modelo, listo para persistir junto a las
    metricas del modelo propuesto y para servir de piso de comparacion.
    """
    import pandas as pd

    filas = [
        evaluar_baseline("media_incondicional",
                         prediccion_media_incondicional(SCORES_STD, T0),
                         SCORES_STD, T0, estandarizador, fpca, X_obs, tau),
        evaluar_baseline(f"persistencia_lag{h}",
                         prediccion_persistencia(SCORES_STD, T0, h),
                         SCORES_STD, T0, estandarizador, fpca, X_obs, tau),
    ]
    return pd.DataFrame(filas).set_index("modelo")
