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
    # -- Bloque B: intervalos de prediccion --
    "winkler",
    "indicador_cobertura",
    "picp",
    "mpiw",
    "resumen_intervalo",
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


# ==========================================================================
# BLOQUE B - INTERVALOS DE PREDICCION
# ==========================================================================
#
# Jerarquia de uso, fijada ANTES de ver resultados
# -----------------------------------------------
# El puntaje de Winkler (interval score) es la unica metrica PRIMARIA de este
# bloque: es una regla de puntuacion PROPIA, de modo que no se puede mejorar
# ensanchando ni estrechando el intervalo, y combina en una cifra el ancho y la
# penalizacion por fallo ponderada por la DISTANCIA a la que quedo la
# observacion. PICP y MPIW son DIAGNOSTICAS y nunca rankean: son la
# descomposicion del Winkler, y su papel es decir, cuando este es malo, si fue
# por cobertura insuficiente o por ancho excesivo. Un intervalo
# arbitrariamente ancho alcanza la cobertura nominal sin informar nada, y por
# eso PICP jamas se reporta sola.
#
# Para quien SI hay intervalos, y para quien no
# ---------------------------------------------
# Este bloque necesita (li, ls) por origen. En el estudio solo el PSBPM-FD los
# tiene de forma nativa --por cuantiles empiricos de su predictiva muestral--;
# el FAR, el RF, el GBT y las lineas base producen predicciones PUNTUALES y no
# una predictiva, de modo que sus celdas del Bloque B quedan VACIAS y asi hay
# que reportarlas. Rellenarlas con una banda gaussiana de residuos dentro de
# muestra las haria parecer artificialmente angostas e inflaria el Winkler de
# las referencias a favor del modelo propuesto: seria una comparacion decidida
# por el supuesto y no por los datos. Dotar a las referencias de una predictiva
# comparable --conformal por bloques sobre un tramo de calibracion, por
# ejemplo-- es una tarea propia y previa, no parte de este bloque.

def winkler(y_obs, li, ls, nivel: float = 0.95) -> np.ndarray:
    """
    Puntaje de Winkler (interval score) elemento a elemento. Menor es mejor.

        IS_alpha(t) = (U_t - L_t)
                      + (2/alpha) (L_t - y_t) 1{y_t < L_t}
                      + (2/alpha) (y_t - U_t) 1{y_t > U_t}

    con alpha = 1 - nivel. La penalizacion crece LINEALMENTE con la distancia a
    la que quedo la observacion, de modo que fallar por poco cuesta poco: eso
    es lo que la separa del indicador de cobertura, que trata igual un fallo
    marginal y uno catastrofico.

    Las formas se preservan: con entradas (n,) devuelve (n,), y con (n, G)
    devuelve (n, G) --el puntaje puntual, un tau a la vez--. Agregarlo sobre el
    dominio es cosa de `resumen_intervalo`, que lo integra con la cuadratura
    comun y no con una suma simple.
    """
    if not 0.0 < nivel < 1.0:
        raise ValueError(f"nivel={nivel} debe estar en (0, 1).")
    y = np.asarray(y_obs, float)
    L = np.asarray(li, float)
    U = np.asarray(ls, float)
    if not (y.shape == L.shape == U.shape):
        raise ValueError(f"Formas incompatibles: y {y.shape}, li {L.shape}, "
                         f"ls {U.shape}.")
    if np.any(U < L):
        raise ValueError("Hay intervalos con ls < li: revisar el orden de los "
                         "cuantiles.")
    alpha = 1.0 - nivel
    return ((U - L)
            + (2.0 / alpha) * np.maximum(L - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - U, 0.0))


def indicador_cobertura(y_obs, li, ls) -> np.ndarray:
    """
    Indicador  I_t = 1{ L_t <= y_t <= U_t }, SIN agregar.

    Se devuelve desagregado a proposito: es el insumo del analisis de
    agrupamiento de las violaciones --si los fallos se concentran en rachas la
    cobertura marginal puede ser nominal y el modelo estar mal calibrado
    condicionalmente-- y de la cobertura por bloque, por horizonte o por
    estrato del generador. `cobertura` promedia; esto no.
    """
    y = np.asarray(y_obs, float)
    L = np.asarray(li, float)
    U = np.asarray(ls, float)
    if not (y.shape == L.shape == U.shape):
        raise ValueError(f"Formas incompatibles: y {y.shape}, li {L.shape}, "
                         f"ls {U.shape}.")
    return ((y >= L) & (y <= U))


def indicador_cobertura_simultanea(y_obs, li, ls) -> np.ndarray:
    """
    Indicador SIMULTANEO por curva: 1 si TODOS los puntos de la curva caen
    dentro de [li, ls]; 0 si al menos uno escapa.

    Distinto de `indicador_cobertura` en que SI agrega -pero sobre el ultimo
    eje (tau u otras componentes de una misma curva), no sobre el tiempo-. Con
    entrada (n, G) el resultado es (n,): una cifra por origen que responde
    "?la curva ENTERA quedo cubierta?", no "?que fraccion de la curva cubre?"
    -que es lo que promedia `indicador_cobertura(...).mean(axis=-1)`, la
    cobertura PUNTUAL de siempre-. Es la version que corresponde a un
    intervalo de credibilidad simultaneo sobre la curva completa: con w
    origenes en una ventana, el promedio de este indicador es la fraccion de
    curvas totalmente contenidas (18/20 curvas -> 0.90), no el promedio de
    fracciones puntuales cubiertas.
    """
    y = np.asarray(y_obs, float)
    L = np.asarray(li, float)
    U = np.asarray(ls, float)
    if not (y.shape == L.shape == U.shape):
        raise ValueError(f"Formas incompatibles: y {y.shape}, li {L.shape}, "
                         f"ls {U.shape}.")
    return ((y >= L) & (y <= U)).all(axis=-1)


def picp(y_obs, li, ls, nivel: float = 0.95) -> dict:
    """
    Cobertura empirica (PICP) con su error estandar y el desvio ACE.

        PICP = (1/n) sum_t I_t        ACE = PICP - nivel

    El error estandar es el binomial, sqrt(p(1-p)/n), y por eso viene con una
    ADVERTENCIA que hay que arrastrar al reporte: supone indicadores
    independientes, y los origenes de una serie no lo son. Con ventanas
    solapadas lo son todavia menos. Es una cota OPTIMISTA del error, util para
    descartar diferencias que ni siquiera lo superan, y para cuantificar en
    serio la incertidumbre hay que usar el bootstrap de bloques de
    `fit/incertidumbre.py`.

    `n_efectivo` cuenta elementos, de modo que con entradas (n, G) son n*G
    pares (origen, tau) y no n curvas: la cobertura es PUNTUAL, no simultanea
    sobre la curva, y el error estandar que sale de ahi es aun mas optimista.
    """
    I = indicador_cobertura(y_obs, li, ls).astype(float)
    n = int(I.size)
    p = float(I.mean())
    return {"picp": p,
            "ee_picp": float(np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))),
            "ace": float(p - nivel),
            "n_efectivo": n,
            "ee_supone_independencia": True}


def mpiw(li, ls) -> float:
    """
    Ancho medio del intervalo (MPIW). Siempre acompana a PICP.

    Por si sola no ordena modelos: un intervalo mas angosto es mejor solo a
    igualdad de cobertura, y sin la cobertura al lado el MPIW premia
    exactamente al modelo sobreconfiado.
    """
    L = np.asarray(li, float)
    U = np.asarray(ls, float)
    if L.shape != U.shape:
        raise ValueError(f"Formas incompatibles: li {L.shape}, ls {U.shape}.")
    if np.any(U < L):
        raise ValueError("Hay intervalos con ls < li.")
    return float((U - L).mean())


def resumen_intervalo(y_obs, li, ls, nivel: float = 0.95,
                      tau: Optional[np.ndarray] = None) -> dict:
    """
    Las cuatro cifras del Bloque B en un dict, con la primaria marcada.

    y_obs, li, ls : (n,) por score, o (n, G) sobre la curva.
    tau : entregado junto a entradas (n, G), el Winkler y el ancho se INTEGRAN
        sobre el dominio con la cuadratura comun --normalizada por la longitud
        del dominio, para que la cifra siga en las unidades de la curva-- y
        despues se promedian sobre los origenes. Sin `tau` se promedia
        elemento a elemento, que con grilla regular difiere solo en el peso 1/2
        de los extremos.

    Claves: `winkler` (PRIMARIA), y `picp`, `ee_picp`, `ace`, `mpiw`
    (DIAGNOSTICAS). El dict lleva `primaria` para que una tabla no pueda
    ordenar modelos por una diagnostica sin haberlo decidido a proposito.
    """
    W = winkler(y_obs, li, ls, nivel=nivel)
    L = np.asarray(li, float)
    U = np.asarray(ls, float)
    anchos = U - L

    if tau is not None and W.ndim == 2:
        w = pesos_trapezoidales(tau)
        dominio = float(w.sum())
        w_val = float(np.mean(integrar(W, tau) / dominio))
        anc = float(np.mean(integrar(anchos, tau) / dominio))
    else:
        w_val = float(W.mean())
        anc = float(anchos.mean())

    d = picp(y_obs, li, ls, nivel=nivel)
    return {"winkler": w_val,
            "picp": d["picp"], "ee_picp": d["ee_picp"], "ace": d["ace"],
            "mpiw": anc, "nivel": float(nivel), "n_efectivo": d["n_efectivo"],
            "primaria": "winkler"}
