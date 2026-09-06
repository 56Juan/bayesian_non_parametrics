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

from ..utils.quadrature import integrar, pesos_trapezoidales

__all__ = [
    "rmse",
    "mse_por_coeficiente",
    "rmse_por_coeficiente",
    "r2_por_columna",
    "razon_dispersion",
    "mise",
    "rmse_funcional",
    "resumen_puntual",
    # -- Bloque A: normas L^p del error --
    "mae",
    "mae_por_coeficiente",
    "cuantil_error_absoluto",
    "error_maximo",
    "pesos_normalizados",
    "normas_error_por_origen",
    "resumen_error_funcional",
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


# ==========================================================================
# BLOQUE A - NORMAS L^p DEL ERROR
# ==========================================================================
#
# El error de prediccion se resume mediante tres normas del MISMO objeto
# e_t(tau) = X_t(tau) - Xhat_t(tau), y no mediante metricas de familias
# distintas. Cada una identifica un funcional distinto de la ley condicional y
# por eso pueden ordenar los modelos de forma distinta:
#
#     ||e||_1  (MAE)  --> mediana condicional. Robusta a valores extremos.
#     ||e||_2  (RMSE) --> media condicional. Es el funcional que el PSBPM-FD
#                         estima, y por eso es la norma coherente con el resto
#                         del capitulo. Penalizacion cuadratica.
#     ||e||_inf       --> peor caso. Depende de UNA sola evaluacion, de modo
#                         que su error de muestreo no baja con n; se reporta
#                         junto al cuantil Q_q(|e|), que mide la misma cola sin
#                         esa fragilidad, y junto al argmax, que dice DONDE
#                         ocurre el peor error.
#
# Las tres cumplen  ||e||_1 <= ||e||_2 <= ||e||_inf  y esa cadena se VERIFICA
# con assert dentro del calculo: si se rompe hay un error de escala o de
# cuadratura y conviene que falle ahi y no aguas abajo como una cifra rara. La
# cadena solo vale con la medida NORMALIZADA a masa uno --Jensen y
# Cauchy-Schwarz exigen una medida de probabilidad--, y esa es la razon de ser
# de `pesos_normalizados`: con los pesos trapezoidales crudos, que suman la
# longitud del dominio, la desigualdad se invierte en cuanto el dominio deja de
# ser [0, 1].
#
# La razon ||e||_inf / ||e||_1 es adimensional y diagnostica CONCENTRACION del
# error: proxima a uno el error esta repartido por todo el dominio; grande, se
# concentra en pocos puntos --un pico, la frontera de la grilla, el arranque de
# la curva-- y ese es un modo de fallo que el MISE promedia y esconde.
#
# NINGUNA de estas metricas esta escalada (no hay MASE ni RMSSE): las
# comparaciones valen DENTRO de una misma serie, no entre series de magnitudes
# distintas. Es una limitacion declarada, no un olvido.

def pesos_normalizados(tau, pesos_tau=None) -> np.ndarray:
    """
    Pesos de la integral sobre el dominio funcional, normalizados a masa uno.

    tau : (G,) grilla del dominio funcional.
    pesos_tau : (G,) ponderacion opcional del dominio. Es el punto de extension
        para pesar unas regiones de tau mas que otras --las horas de crecida de
        una serie diaria, por ejemplo--; se multiplica por la cuadratura, de
        modo que `None` es la ponderacion uniforme y reproduce exactamente la
        trapezoidal de siempre. Debe ser LA MISMA para todos los modelos que se
        comparan, y por eso viaja como parametro explicito y no como decision
        interna de cada metrica.

    Ojo con la otra ponderacion del estudio: esta pesa el DOMINIO FUNCIONAL
    tau, y no es la que agregaria sobre HORIZONTES de prediccion. Son dos
    objetos distintos y se nombran distinto a proposito; hoy el estudio corre
    con un solo horizonte (h=1, con los rezagos reales en t), de modo que la
    segunda no existe todavia en el codigo.
    """
    w = pesos_trapezoidales(np.asarray(tau, dtype=float))
    if pesos_tau is not None:
        v = np.asarray(pesos_tau, dtype=float).ravel()
        if v.size != w.size:
            raise ValueError(f"pesos_tau tiene {v.size} puntos y tau {w.size}.")
        if np.any(v < 0):
            raise ValueError("pesos_tau no puede tener componentes negativas.")
        w = w * v
    masa = float(w.sum())
    if not np.isfinite(masa) or masa <= 0:
        raise ValueError("Los pesos del dominio suman cero o no son finitos.")
    return w / masa


def mae(y_obs, y_pred) -> float:
    """Error absoluto medio sobre todos los elementos (norma L^1 discreta)."""
    y_obs, y_pred = np.asarray(y_obs, float), np.asarray(y_pred, float)
    if y_obs.shape != y_pred.shape:
        raise ValueError(f"Formas incompatibles: {y_obs.shape} vs {y_pred.shape}.")
    return float(np.abs(y_pred - y_obs).mean())


def mae_por_coeficiente(alpha_obs, alpha_pred) -> np.ndarray:
    """MAE por columna; el analogo L^1 de `mse_por_coeficiente`."""
    A, P = _2d(alpha_obs), _2d(alpha_pred)
    if A.shape != P.shape:
        raise ValueError(f"Formas incompatibles: {A.shape} vs {P.shape}.")
    return np.abs(P - A).mean(axis=0)


def cuantil_error_absoluto(y_obs, y_pred, q: float = 0.95) -> float:
    """
    Cuantil `q` de |e| sobre todos los elementos.

    Es la cifra que se reporta EN LUGAR del supremo cuando interesa el peor
    caso: el supremo lo decide una sola observacion y su error de muestreo no
    baja con n, mientras que el cuantil 0.95 mide la misma cola y es estable.
    """
    y_obs, y_pred = np.asarray(y_obs, float), np.asarray(y_pred, float)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q={q} debe estar en (0, 1).")
    return float(np.quantile(np.abs(y_pred - y_obs), q))


def error_maximo(y_obs, y_pred) -> dict:
    """
    Supremo de |e| y la posicion donde ocurre.

    El argmax no es un adorno: dice si el peor error esta en la frontera del
    dominio --tipico del truncamiento de la base--, en un origen concreto --un
    episodio-- o repartido.
    """
    y_obs, y_pred = np.asarray(y_obs, float), np.asarray(y_pred, float)
    if y_obs.shape != y_pred.shape:
        raise ValueError(f"Formas incompatibles: {y_obs.shape} vs {y_pred.shape}.")
    E = np.abs(y_pred - y_obs)
    i = int(np.argmax(E))
    return {"linf": float(E.flat[i]), "argmax_plano": i,
            "argmax": tuple(int(v) for v in np.unravel_index(i, E.shape))}


def normas_error_por_origen(X_obs, X_pred, tau, pesos_tau=None,
                            q: float = 0.95, verificar: bool = True) -> dict:
    """
    Las tres normas del error funcional, UNA CIFRA POR ORIGEN.

    X_obs, X_pred : (n, G). En simulacion `X_obs` son las curvas VERDADERAS del
        generador: sigma_obs no es algo que el modelo deba predecir.
    tau : (G,) grilla. Cuadratura comun del proyecto, no una suma simple.
    pesos_tau : ponderacion opcional del dominio (ver `pesos_normalizados`).
    q : orden del cuantil de |e| sobre el dominio, por origen.
    verificar : comprueba la cadena L^1 <= L^2 <= L^inf origen a origen.

    Retorna un dict de arreglos (n,):
        l1, l2, linf   las tres normas
        q_abs          cuantil q de |e_t| sobre la grilla
        argmax_tau     el tau donde ocurre el maximo, por origen
        razon_linf_l1  concentracion del error, adimensional

    Las metricas del Bloque A se reportan como CURVA sobre los origenes --de
    ahi que esta funcion devuelva vectores y no escalares--; `resumen_error_funcional`
    agrega, y es ahi donde el orden de agregacion importa y hay que declararlo.
    """
    O, P = _2d(X_obs), _2d(X_pred)
    if O.shape != P.shape:
        raise ValueError(f"Formas incompatibles: {O.shape} vs {P.shape}.")
    tau = np.asarray(tau, dtype=float).ravel()
    if tau.size != O.shape[1]:
        raise ValueError(f"tau tiene {tau.size} puntos y las curvas {O.shape[1]}.")

    w = pesos_normalizados(tau, pesos_tau)
    E = np.abs(P - O)                                   # (n, G)

    l1 = E @ w
    l2 = np.sqrt((E ** 2) @ w)
    i_max = np.argmax(E, axis=1)
    linf = E[np.arange(E.shape[0]), i_max]
    q_abs = np.quantile(E, q, axis=1)

    if verificar:
        verificar_cadena_lp(l1, l2, linf)

    return {"l1": l1, "l2": l2, "linf": linf, "q_abs": q_abs,
            "argmax_tau": tau[i_max], "argmax_idx": i_max.astype(int),
            "razon_linf_l1": linf / np.maximum(l1, 1e-300),
            "q": float(q)}


def verificar_cadena_lp(l1, l2, linf, tol: float = 1e-9) -> None:
    """
    Verifica  ||e||_1 <= ||e||_2 <= ||e||_inf  origen a origen.

    Se comprueba en el calculo y no en un comentario: la cadena es una
    identidad matematica bajo una medida de probabilidad, de modo que si falla
    no hay nada que interpretar --hay un error de escala, de cuadratura o de
    normalizacion de los pesos-- y conviene que reviente aqui, con el origen
    culpable en el mensaje.

    `tol` es absoluta y minuscula a proposito: lo unico que se tolera es el
    redondeo de punto flotante.
    """
    l1 = np.asarray(l1, float).ravel()
    l2 = np.asarray(l2, float).ravel()
    linf = np.asarray(linf, float).ravel()
    mal_12 = np.where(l1 > l2 + tol)[0]
    mal_2i = np.where(l2 > linf + tol)[0]
    if mal_12.size or mal_2i.size:
        i = int(mal_12[0]) if mal_12.size else int(mal_2i[0])
        raise AssertionError(
            "Se rompe la cadena ||e||_1 <= ||e||_2 <= ||e||_inf en el origen "
            f"{i}: l1={l1[i]:.6g}, l2={l2[i]:.6g}, linf={linf[i]:.6g}. Es una "
            "identidad bajo una medida de masa uno: revisar la normalizacion "
            "de los pesos del dominio o la cuadratura.")


def resumen_error_funcional(X_obs, X_pred, tau, pesos_tau=None,
                            q: float = 0.95) -> dict:
    """
    Agrega sobre los origenes las normas de `normas_error_por_origen`.

    ORDEN DE AGREGACION, que es justo la ambiguedad que hay que cerrar
    ------------------------------------------------------------------
    Promediar RMSE por origen NO es la raiz del MSE agregado: por Jensen,
    mean_t ||e_t||_2 <= sqrt( mean_t ||e_t||_2^2 ), con igualdad solo si el
    error es constante entre origenes. Las dos cantidades son legitimas y
    responden a preguntas distintas, de modo que la unica salida honesta es
    declarar cual se usa. Aqui:

        rmse_f   = sqrt( mean_t int e_t^2 dw )   <- RAIZ DEL MSE AGREGADO.
                   Integra primero sobre tau, promedia despues sobre t y toma
                   la raiz al final. Es la convencion del resto del capitulo:
                   coincide con `sqrt(mise(...))` y con el `rmse_f` que emite
                   `ventana_movil_funcional`, de modo que las cifras de este
                   bloque son directamente comparables con las tablas 51 y 57
                   que ya existen.

        l2_medio = mean_t ||e_t||_2              <- PROMEDIO DE RMSE por origen.
                   Se reporta AL LADO, no en lugar de la anterior. Su razon con
                   `rmse_f` (`razon_agregacion`, siempre <= 1) mide cuan
                   desigual es el error entre origenes: proxima a uno el error
                   es homogeneo en el tiempo; baja, hay episodios que dominan
                   el agregado y el promedio simple los diluye.

    El MAE no tiene esta ambiguedad --es lineal, luego aditivo sobre origenes y
    sobre el dominio-- y esa es una de sus ventajas practicas.

    La norma L^inf se agrega de dos formas por la misma razon: `linf_max` es el
    peor caso global (una sola evaluacion, fragil) y `linf_medio` el peor caso
    tipico por origen. La cifra CITABLE es la del cuantil.
    """
    d = normas_error_por_origen(X_obs, X_pred, tau, pesos_tau=pesos_tau, q=q)
    i_peor = int(np.argmax(d["linf"]))
    rmse_f = float(np.sqrt((d["l2"] ** 2).mean()))
    return {
        "mae_f":      float(d["l1"].mean()),
        "rmse_f":     rmse_f,
        "l2_medio":   float(d["l2"].mean()),
        "razon_agregacion": float(d["l2"].mean() / max(rmse_f, 1e-300)),
        "linf_max":   float(d["linf"].max()),
        "linf_medio": float(d["linf"].mean()),
        f"q{int(round(q * 100))}_abs_medio": float(d["q_abs"].mean()),
        "razon_linf_l1_media": float(d["razon_linf_l1"].mean()),
        "argmax_tau_peor":     float(d["argmax_tau"][i_peor]),
        "argmax_origen_peor":  i_peor,
        "n": int(d["l1"].size),
    }
