"""
metrics_distribucional.py
=========================
Evaluacion de la distribucion predictiva (Seccion 2.2.3.3).

Un modelo que estima la distribucion predictiva completa debe evaluarse en la
calidad de dicha distribucion y no solo en su prediccion puntual. Se emplean
reglas de puntuacion propias, cuyo valor esperado se optimiza unicamente
cuando la distribucion declarada coincide con la verdadera, de modo que ningun
modelo puede mejorar su puntaje declarando una distribucion distinta de la que
efectivamente estima.

Convencion de formas
--------------------
Las funciones basadas en muestras reciben `muestras` con forma (S, n) o
(S, n, G), donde S indexa las extracciones de la distribucion predictiva y las
dimensiones restantes los origenes de prediccion y, en su caso, los puntos de
la grilla. Esta convencion coincide con la salida natural de un muestreador
MCMC tras el descarte inicial.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..utils.quadrature import integrar, pesos_trapezoidales

__all__ = [
    "crps_muestral",
    "crps_gaussiano",
    "energy_score",
    "cobertura",
    "estratos_por_cuantil",
    "cobertura_condicional",
    "intervalo_muestral",
    "pit_muestral",
    "pit_gaussiano",
    "diagnostico_pit",
    "lps_gaussiano",
    "lps_desde_log_densidad",
]


# ==========================================================================
# CRPS
# ==========================================================================

def crps_muestral(y_obs: np.ndarray, muestras: np.ndarray) -> np.ndarray:
    """
    Puntaje de probabilidad de rango continuo estimado a partir de muestras.

    Emplea la representacion

        CRPS(F, y) = E|Z - y| - (1/2) E|Z - Z'|,   Z, Z' iid ~ F,

    donde el primer termino mide la distancia esperada entre la predictiva y la
    observacion y el segundo la dispersion interna de la predictiva.

    y_obs    : (n,)
    muestras : (S, n)
    Retorna  : (n,) puntaje por origen. Menor es mejor.

    El segundo termino se calcula con el estimador basado en el ordenamiento de
    las muestras, de costo O(S log S) por origen en lugar del O(S^2) que exige
    la doble suma directa.
    """
    y = np.asarray(y_obs, dtype=float).ravel()
    Z = np.asarray(muestras, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != y.size:
        raise ValueError(
            f"muestras debe tener forma (S, n) con n={y.size}; recibido {Z.shape}.")
    S = Z.shape[0]
    if S < 2:
        raise ValueError("Se requieren al menos 2 muestras para estimar el CRPS.")

    term1 = np.mean(np.abs(Z - y[None, :]), axis=0)

    Zs = np.sort(Z, axis=0)
    pesos = (2.0 * np.arange(1, S + 1) - S - 1).astype(float)[:, None]
    term2 = 2.0 * np.sum(pesos * Zs, axis=0) / (S * (S - 1))

    return term1 - 0.5 * term2


def crps_gaussiano(y_obs, mu, sd) -> np.ndarray:
    """
    CRPS en forma cerrada bajo predictiva gaussiana.

    Util como referencia y para la aproximacion de dos momentos, pero notese
    que la predictiva del modelo de mezcla no es gaussiana: aplicar esta forma
    a una predictiva multimodal subestima el puntaje real y anula justamente la
    ventaja que la mezcla busca capturar.
    """
    from math import pi
    y = np.asarray(y_obs, float).ravel()
    mu = np.asarray(mu, float).ravel()
    sd = np.asarray(sd, float).ravel()
    z = (y - mu) / np.where(sd > 0, sd, np.nan)
    Phi = 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * pi)
    return sd * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(pi))


def _erf(x: np.ndarray) -> np.ndarray:
    """Funcion error vectorizada sin dependencia de SciPy."""
    from math import erf as _e
    return np.vectorize(_e)(np.asarray(x, dtype=float))


# ==========================================================================
# ENERGY SCORE
# ==========================================================================

def energy_score(X_obs: np.ndarray, muestras: np.ndarray,
                 max_pares: int = 2000, seed: int = 0) -> float:
    """
    Puntaje de energia, generalizacion multivariada del CRPS.

    Trata la curva evaluada en la grilla como un vector, con lo que resulta
    sensible a la estructura de dependencia entre puntos del dominio que el
    CRPS integrado no captura.

    X_obs    : (n, G) curvas observadas
    muestras : (S, n, G) extracciones de la predictiva por origen
    Retorna el puntaje promediado sobre los origenes. Menor es mejor.

    El segundo termino se estima sobre un subconjunto aleatorio de pares
    cuando S es grande, dado que su calculo exacto es cuadratico en S.
    """
    O = np.atleast_2d(np.asarray(X_obs, dtype=float))
    Z = np.asarray(muestras, dtype=float)
    if Z.ndim != 3 or Z.shape[1:] != O.shape:
        raise ValueError(
            f"muestras debe tener forma (S, n, G) con (n, G)={O.shape}; "
            f"recibido {Z.shape}.")
    S = Z.shape[0]
    if S < 2:
        raise ValueError("Se requieren al menos 2 muestras.")

    term1 = np.mean(np.linalg.norm(Z - O[None, :, :], axis=2), axis=0)

    rng = np.random.default_rng(seed)
    n_pares = min(max_pares, S * (S - 1) // 2)
    i = rng.integers(0, S, size=n_pares)
    j = rng.integers(0, S, size=n_pares)
    valido = i != j
    i, j = i[valido], j[valido]
    term2 = np.mean(np.linalg.norm(Z[i] - Z[j], axis=2), axis=0)

    return float(np.mean(term1 - 0.5 * term2))


# ==========================================================================
# COBERTURA
# ==========================================================================

def intervalo_muestral(muestras: np.ndarray, nivel: float = 0.95):
    """Cuantiles empiricos simetricos de la predictiva; retorna (li, ls)."""
    Z = np.asarray(muestras, dtype=float)
    alpha = (1.0 - nivel) / 2.0
    return (np.quantile(Z, alpha, axis=0), np.quantile(Z, 1.0 - alpha, axis=0))


def cobertura(y_obs, li, ls, tau: Optional[np.ndarray] = None) -> dict:
    """
    Cobertura empirica puntual y ancho medio del intervalo.

    Con `tau` entregado y entradas de forma (n, G), la cobertura se promedia
    sobre el dominio ademas de sobre los origenes, conforme a la definicion
    funcional de la Seccion 2.2.3.3.

    Una cobertura proxima al nivel nominal es condicion necesaria de
    calibracion pero no suficiente, puesto que intervalos arbitrariamente
    anchos la alcanzan sin aportar informacion; por ello se reporta junto al
    ancho medio, y entre modelos con cobertura comparable se prefiere el de
    intervalos mas angostos.
    """
    y = np.asarray(y_obs, float)
    li = np.asarray(li, float)
    ls = np.asarray(ls, float)
    dentro = ((y >= li) & (y <= ls)).astype(float)
    ancho = ls - li

    if tau is not None and y.ndim == 2:
        w = pesos_trapezoidales(tau)
        dominio = float(w.sum())
        cob = float(np.mean(integrar(dentro, tau) / dominio))
        anc = float(np.mean(integrar(ancho, tau) / dominio))
    else:
        cob = float(dentro.mean())
        anc = float(ancho.mean())

    return {"cobertura": cob, "ancho_medio": anc}


# ==========================================================================
# COBERTURA CONDICIONAL AL ESTADO DEL GENERADOR
# ==========================================================================
#
# La cobertura marginal de `cobertura` promedia sobre todos los origenes y por
# ello no distingue un modelo calibrado de uno que compensa: bandas demasiado
# anchas en los periodos de calma contra bandas demasiado angostas en los de
# estres promedian al nivel nominal y superan el control. El diseno del estudio
# (docs/03 Modelo.tex, seccion 03_06, eje 2) pide en consecuencia estratificar
# los origenes segun el ESTADO VERDADERO del generador --nivel de volatilidad
# en el Algoritmo 2, regimen activo en el Algoritmo 3-- y examinar la cobertura
# dentro de cada estrato.
#
# El estado verdadero es una cantidad latente que solo el generador conoce:
# `SalidaSimulacion.internos` la expone (`sigma2` en el Algoritmo 2,
# `regimenes` en el Algoritmo 3). No debe reemplazarse por un proxy estimado a
# partir de los datos --por ejemplo la varianza empirica en una ventana-- sin
# declararlo, porque entonces la estratificacion depende del ruido de medicion
# y del ancho de la ventana, y deja de ser una particion exogena.

def estratos_por_cuantil(valores: np.ndarray, n_estratos: int = 3,
                         etiquetas: Optional[list] = None):
    """
    Particiona los origenes en `n_estratos` grupos por cuantiles de `valores`.

    Devuelve `(indice, etiquetas)`, con `indice` de forma (n,) y valores en
    {0, ..., n_estratos-1}, ordenados de menor a mayor.

    Se emplean cuantiles y no cortes fijos para que los estratos queden
    equilibrados con independencia de la escala del estado, que cambia entre
    escenarios y entre replicas. Cuando la variable de estado es discreta --el
    regimen del Algoritmo 3-- esta funcion no corresponde: alli el estrato es
    el propio valor del estado y se pasa directamente a `cobertura_condicional`.
    """
    v = np.asarray(valores, dtype=float).ravel()
    if n_estratos < 2:
        raise ValueError("n_estratos debe ser al menos 2.")
    if v.size < n_estratos:
        raise ValueError(
            f"Hay {v.size} origenes y se piden {n_estratos} estratos.")

    cortes = np.quantile(v, np.linspace(0.0, 1.0, n_estratos + 1)[1:-1])
    idx = np.searchsorted(cortes, v, side="right").astype(int)

    if etiquetas is None:
        if n_estratos == 3:
            etiquetas = ["baja", "media", "alta"]
        else:
            etiquetas = [f"q{i + 1}" for i in range(n_estratos)]
    if len(etiquetas) != n_estratos:
        raise ValueError(
            f"Se entregaron {len(etiquetas)} etiquetas para {n_estratos} estratos.")
    return idx, list(etiquetas)


def cobertura_condicional(y_obs, li, ls, estratos,
                          etiquetas: Optional[list] = None,
                          tau: Optional[np.ndarray] = None,
                          muestras: Optional[np.ndarray] = None) -> list:
    """
    Cobertura y ancho medio dentro de cada estrato del estado verdadero.

    Parametros
    ----------
    y_obs, li, ls : (n,) para un score, o (n, G) para la curva.
    estratos : (n,) indice entero de estrato por origen, tal como lo devuelve
        `estratos_por_cuantil` o directamente el estado discreto del generador.
    etiquetas : nombre de cada estrato, en el orden de sus indices.
    tau : grilla; con entradas (n, G) promedia la cobertura sobre el dominio
        con la cuadratura trapezoidal, igual que `cobertura`.
    muestras : (S, n) o (S, n, G), opcional. Si se entrega se agrega el puntaje
        propio del estrato --CRPS para un score, puntaje de energia para la
        curva--, de modo que la lectura no dependa solo de la cobertura: un
        estrato puede alcanzar el nivel nominal con bandas desmedidas y el
        puntaje propio lo penaliza.

    Devuelve una lista de diccionarios, uno por estrato, apta para
    `pd.DataFrame(...)`. Se reporta ademas `desvio`, la diferencia entre la
    cobertura del estrato y la marginal: es la cifra que hace visible la
    compensacion que la marginal esconde.
    """
    y = np.asarray(y_obs, float)
    li = np.asarray(li, float)
    ls = np.asarray(ls, float)
    s = np.asarray(estratos).ravel()

    if s.size != y.shape[0]:
        raise ValueError(
            f"estratos tiene {s.size} entradas y hay {y.shape[0]} origenes.")

    marginal = cobertura(y, li, ls, tau=tau)["cobertura"]
    unicos = np.unique(s)
    if etiquetas is None:
        etiquetas = [f"estrato_{int(u)}" for u in unicos]
    if len(etiquetas) != unicos.size:
        raise ValueError(
            f"Se entregaron {len(etiquetas)} etiquetas para {unicos.size} estratos.")

    filas = []
    for etq, u in zip(etiquetas, unicos):
        m = (s == u)
        cob = cobertura(y[m], li[m], ls[m], tau=tau)
        fila = {
            "estrato": etq,
            "n": int(m.sum()),
            "cobertura": cob["cobertura"],
            "ancho_medio": cob["ancho_medio"],
            "desvio": cob["cobertura"] - marginal,
        }
        if muestras is not None:
            Z = np.asarray(muestras, float)[:, m, ...]
            if y.ndim == 1:
                fila["crps"] = float(crps_muestral(y[m], Z).mean())
            else:
                fila["energy"] = float(energy_score(y[m], Z))
        filas.append(fila)
    return filas


# ==========================================================================
# TRANSFORMADA INTEGRAL DE PROBABILIDAD
# ==========================================================================

def pit_muestral(y_obs, muestras) -> np.ndarray:
    """
    PIT estimada como la fraccion de muestras que no superan la observacion.

    Si la predictiva esta bien calibrada los valores se distribuyen
    aproximadamente uniformes en [0, 1].
    """
    y = np.asarray(y_obs, float)
    Z = np.asarray(muestras, float)
    if Z.shape[1:] != y.shape:
        raise ValueError(f"Formas incompatibles: muestras {Z.shape}, y {y.shape}.")
    return np.mean(Z <= y[None, ...], axis=0)


def pit_gaussiano(y_obs, mu, sd) -> np.ndarray:
    """PIT bajo la aproximacion gaussiana de dos momentos."""
    y = np.asarray(y_obs, float)
    mu = np.asarray(mu, float)
    sd = np.asarray(sd, float)
    z = (y - mu) / np.where(sd > 0, sd, np.nan)
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def diagnostico_pit(u: np.ndarray, n_bins: int = 10) -> dict:
    """
    Resume la desviacion de la PIT respecto de la uniformidad.

    Las desviaciones sistematicas son diagnosticas: un histograma en forma de U
    indica subdispersion de la predictiva y una concentracion central indica
    sobredispersion. Se reporta el histograma normalizado, el estadistico de
    Kolmogorov-Smirnov contra la uniforme y un indicador de forma construido
    como la diferencia entre la masa de los bins extremos y la de los
    centrales, positivo bajo subdispersion y negativo bajo sobredispersion.
    """
    u = np.asarray(u, dtype=float).ravel()
    u = u[np.isfinite(u)]
    if u.size == 0:
        raise ValueError("No hay valores finitos en la PIT.")

    hist, bordes = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    frec = hist / u.size
    esperado = 1.0 / n_bins

    us = np.sort(u)
    emp = np.arange(1, us.size + 1) / us.size
    ks = float(np.max(np.abs(emp - us)))

    n_ext = max(1, n_bins // 5)
    masa_ext = float(frec[:n_ext].sum() + frec[-n_ext:].sum())
    masa_ctr = float(frec[n_bins // 2 - n_ext // 2:
                          n_bins // 2 + max(1, n_ext - n_ext // 2)].sum())

    if masa_ext > 2.5 * n_ext * esperado:
        forma = "U (subdispersion: intervalos demasiado angostos)"
    elif masa_ctr > 2.5 * n_ext * esperado:
        forma = "campana (sobredispersion: intervalos demasiado anchos)"
    else:
        forma = "aproximadamente uniforme"

    return {"n": int(u.size), "hist": frec, "bordes": bordes,
            "ks": ks, "masa_extremos": masa_ext, "masa_central": masa_ctr,
            "forma": forma}


# ==========================================================================
# PUNTAJE LOGARITMICO
# ==========================================================================

def lps_gaussiano(y_obs, mu, sd) -> float:
    """
    Puntaje logaritmico bajo predictiva gaussiana (aproximacion de dos momentos).

    Advertencia: el puntaje logaritmico es sensible a la forma completa de la
    distribucion, que es precisamente lo que esta aproximacion descarta.
    Emplearla para comparar un modelo de mezcla contra uno gaussiano elimina la
    diferencia que se busca medir. Se ofrece como referencia y para modelos
    cuya predictiva es efectivamente gaussiana; para el modelo de mezcla debe
    usarse `lps_desde_log_densidad` con la densidad de la mezcla.
    """
    from math import log, pi
    y = np.asarray(y_obs, float).ravel()
    mu = np.asarray(mu, float).ravel()
    sd = np.asarray(sd, float).ravel()
    ll = -0.5 * np.log(2.0 * pi) - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2
    return float(-np.mean(ll))


def lps_desde_log_densidad(y_obs, log_dens: Callable[[np.ndarray], np.ndarray]) -> float:
    """
    Puntaje logaritmico a partir de una densidad predictiva evaluable.

    `log_dens` recibe las observaciones y retorna el logaritmo de la densidad
    predictiva en cada una. Para el modelo de mezcla esa densidad se obtiene
    promediando sobre las extracciones del posterior la mezcla de nucleos
    ponderada por los pesos, forma cerrada que evita estimar la densidad por
    remuestreo.
    """
    y = np.asarray(y_obs, dtype=float)
    ld = np.asarray(log_dens(y), dtype=float)
    if ld.shape != y.shape:
        raise ValueError(
            f"log_dens debe retornar la forma de y ({y.shape}); recibido {ld.shape}.")
    return float(-np.mean(ld))
