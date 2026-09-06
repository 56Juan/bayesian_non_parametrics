r"""
incertidumbre.py
================
Cuantificacion de la incertidumbre de una metrica agregada sobre origenes
DEPENDIENTES, mediante bootstrap de bloques.

Por que un modulo propio y no `pooling.py`
------------------------------------------
`pooling.py` agrupa CADENAS MCMC: su objeto es la mezcla de igual peso de las
predictivas por cadena, es decir incertidumbre del POSTERIOR, y su unidad de
remuestreo seria la extraccion. Aqui la unidad es el ORIGEN de prediccion y lo
que se cuantifica es el error de MUESTREO de una metrica evaluada sobre una
serie temporal finita. Son dos preguntas distintas sobre dos poblaciones
distintas, y meterlas en el mismo archivo invitaria a mezclar sus unidades de
remuestreo, que es el error clasico en este tipo de codigo. De ahi el modulo
aparte.

Por que de BLOQUES y no ordinario
---------------------------------
El bootstrap ordinario remuestrea observaciones de forma independiente y por
tanto supone que lo son. Los errores de prediccion de una serie de tiempo no lo
son --un episodio de volatilidad, un cambio de regimen o una tendencia mal
seguida producen rachas de error grande-- y, cuando la cifra que se remuestrea
es la de una VENTANA MOVIL, la dependencia es ademas mecanica: dos ventanas
solapadas comparten w-1 de sus w origenes y son casi el mismo numero. Bajo esa
dependencia el bootstrap ordinario subestima la varianza, y lo hace tanto mas
cuanto mayor sea el solapamiento: es decir, falla mas justo donde mas se lo
usaria.

El bootstrap de bloques (Kunsch, 1989) remuestrea TRAMOS CONTIGUOS de largo
`largo_bloque` en vez de puntos sueltos, de modo que la dependencia dentro del
tramo se conserva. Se implementa la variante CIRCULAR (Politis & Romano, 1992):
la serie se cierra en un anillo y los bloques pueden cruzar el extremo. Esto
evita que las observaciones de los bordes se muestreen menos que las del centro
--el sesgo del bootstrap de bloques movil-- a costa de unir artificialmente el
final con el principio de la serie. Con series no estacionarias, que es el caso
de buena parte de los escenarios del estudio, esa union es un supuesto real y
hay que declararlo: el intervalo describe la variabilidad de la metrica bajo la
ley empirica de la serie, no bajo la ley del proceso.

Como elegir `largo_bloque`, y por que no hay un valor por defecto bueno
----------------------------------------------------------------------
La regla operativa del estudio tiene dos partes, y la primera domina:

  1. SOLAPAMIENTO MECANICO. Si lo que se remuestrea son las cifras de una
     ventana movil de ancho `w` deslizada de a `paso`, dos cifras separadas por
     menos de `w/paso` posiciones comparten origenes. El bloque debe cubrir al
     menos ese tramo, o el remuestreo cortara por dentro de una dependencia que
     es aritmetica y no estadistica. `largo_bloque_sugerido` implementa
     exactamente esto.

  2. DEPENDENCIA DEL PROCESO. Si lo que se remuestrea son errores POR ORIGEN
     --sin ventana de por medio-- el bloque tiene que cubrir la memoria de la
     serie de errores. La referencia habitual es del orden de n^(1/3), y una
     comprobacion barata es mirar la autocorrelacion de la serie remuestreada:
     `diagnostico_dependencia` la reporta.

Cuando ambas aplican, manda la mayor. Y `largo_bloque` es un PARAMETRO
explicito, sin valor por defecto oculto, porque la anchura del intervalo
resultante depende de el de forma directa: reportar un intervalo bootstrap sin
decir con que largo de bloque se calculo no es reportar nada.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "largo_bloque_sugerido",
    "bloques_circulares",
    "bootstrap_bloques",
    "diagnostico_dependencia",
    "tabla_bootstrap",
]


def largo_bloque_sugerido(n: int, w: Optional[int] = None, paso: int = 1) -> dict:
    """
    Largo de bloque sugerido, con las dos reglas explicitas y la que manda.

    n : numero de cifras que se van a remuestrear.
    w : ancho de la ventana movil que produjo esas cifras, si las produjo una.
        Con `None` se supone que son errores por origen, sin ventana.
    paso : deslizamiento de la ventana.

    Retorna las dos sugerencias y `largo_bloque`, que es el maximo de ambas
    acotado a n. Es una SUGERENCIA: el valor con que se reporta se declara.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"n={n} es demasiado corto para remuestrear.")
    por_memoria = max(1, int(round(n ** (1.0 / 3.0))))
    por_solape = 1 if w is None else max(1, int(np.ceil(int(w) / max(int(paso), 1))))
    elegido = min(n, max(por_memoria, por_solape))
    return {"largo_bloque": int(elegido),
            "por_memoria_n13": int(por_memoria),
            "por_solape_ventana": int(por_solape),
            "manda": ("solape de la ventana movil" if por_solape >= por_memoria
                      else "memoria de la serie (n^(1/3))"),
            "n": n}


def bloques_circulares(n: int, largo_bloque: int, rng) -> np.ndarray:
    """
    Indices (base-0) de una replica del bootstrap de bloques circular.

    Se sortean ceil(n / largo_bloque) puntos de arranque uniformes en {0..n-1},
    se toma de cada uno un tramo contiguo modulo n y se recorta a largo n. El
    cierre en anillo es lo que iguala la probabilidad de aparecer de todas las
    observaciones, bordes incluidos.
    """
    n, b = int(n), int(largo_bloque)
    if not 1 <= b <= n:
        raise ValueError(f"largo_bloque={b} fuera de [1, {n}].")
    n_bloques = int(np.ceil(n / b))
    arranques = rng.integers(0, n, size=n_bloques)
    idx = (arranques[:, None] + np.arange(b)[None, :]).ravel() % n
    return idx[:n]


def bootstrap_bloques(valores, estadistico: Callable[[np.ndarray], float] = np.mean,
                      largo_bloque: Optional[int] = None,
                      B: int = 2000, nivel: float = 0.95,
                      w: Optional[int] = None, paso: int = 1,
                      seed: int = 0) -> dict:
    """
    Intervalo bootstrap de bloques para un estadistico de una serie dependiente.

    valores : (n,) la serie a remuestrear, en el ORDEN TEMPORAL. Puede ser el
        error por origen, la cifra de cada ventana movil, el Winkler por origen
        o el indicador de cobertura --con `np.mean` como estadistico, ese
        ultimo caso da el intervalo de la PICP que el error estandar binomial
        de `picp` subestima.
    estadistico : f(muestra) -> float. Por defecto la media. Para el RMSE
        agregado hay que pasar el estadistico sobre los errores CUADRATICOS
        (`lambda e2: np.sqrt(e2.mean())`) y no promediar RMSE: es la misma
        distincion de orden de agregacion que documenta `resumen_error_funcional`.
    largo_bloque : si es None se toma el de `largo_bloque_sugerido(n, w, paso)`
        y el valor efectivamente usado viaja en el resultado. NO se elige a
        ciegas: ver el encabezado del modulo.
    B : numero de replicas.
    nivel : nivel del intervalo percentil.
    seed : semilla del generador.

    Retorna el valor observado, la media y la sd bootstrap, el intervalo
    percentil y el largo de bloque usado. El intervalo es PERCENTIL simple, sin
    correccion BCa: con B >= 2000 y estadisticos suaves alcanza para lo que se
    usa aqui --decidir si dos modelos se distinguen-- y la correccion añadiria
    una capa cuyo supuesto tampoco se verifica bajo dependencia.
    """
    v = np.asarray(valores, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        raise ValueError(f"Quedan {n} valores finitos: nada que remuestrear.")
    if not 0.0 < nivel < 1.0:
        raise ValueError(f"nivel={nivel} debe estar en (0, 1).")

    sug = largo_bloque_sugerido(n, w=w, paso=paso)
    b = int(largo_bloque) if largo_bloque is not None else sug["largo_bloque"]
    if not 1 <= b <= n:
        raise ValueError(f"largo_bloque={b} fuera de [1, {n}].")

    rng = np.random.default_rng(seed)
    reps = np.empty(int(B), dtype=float)
    for r in range(int(B)):
        reps[r] = float(estadistico(v[bloques_circulares(n, b, rng)]))

    alpha = (1.0 - nivel) / 2.0
    return {
        "observado":    float(estadistico(v)),
        "media_boot":   float(reps.mean()),
        "sd_boot":      float(reps.std(ddof=1)),
        "li":           float(np.quantile(reps, alpha)),
        "ls":           float(np.quantile(reps, 1.0 - alpha)),
        "nivel":        float(nivel),
        "largo_bloque": int(b),
        "largo_bloque_sugerido": int(sug["largo_bloque"]),
        "criterio_bloque": sug["manda"] if largo_bloque is None else "declarado",
        "B": int(B), "n": int(n),
    }


def diagnostico_dependencia(valores, k_max: int = 5) -> dict:
    """
    Autocorrelacion de la serie remuestreada, para justificar `largo_bloque`.

    Si `acf1` es alta, un bloque corto no captura la dependencia y el intervalo
    quedara demasiado angosto. Con cifras de ventana movil solapada la acf es
    alta POR CONSTRUCCION y no dice nada sobre el proceso: ahi el criterio que
    manda es el del solapamiento, no este.
    """
    v = np.asarray(valores, dtype=float).ravel()
    v = v[np.isfinite(v)]
    c = v - v.mean()
    den = float((c ** 2).sum())
    acf = [float((c[k:] * c[:-k]).sum() / den) if den > 0 else np.nan
           for k in range(1, int(k_max) + 1)]
    return {"n": int(v.size),
            "acf1": acf[0] if acf else np.nan,
            "acf": acf,
            "primer_rezago_bajo_0.1": next(
                (k + 1 for k, a in enumerate(acf) if abs(a) < 0.1), None)}


def tabla_bootstrap(series: dict, estadistico: Callable = np.mean,
                    largo_bloque: Optional[int] = None,
                    B: int = 2000, nivel: float = 0.95,
                    w: Optional[int] = None, paso: int = 1,
                    seed: int = 0) -> pd.DataFrame:
    """
    Aplica `bootstrap_bloques` a un dict {etiqueta: serie} y devuelve una tabla.

    Todas las series comparten `largo_bloque`, `B`, `nivel` y semilla: si cada
    una usara el suyo, la anchura de los intervalos ya no seria comparable
    entre modelos, que es justamente para lo que se los mira.
    """
    filas = []
    for nombre, v in series.items():
        d = bootstrap_bloques(v, estadistico=estadistico,
                              largo_bloque=largo_bloque, B=B, nivel=nivel,
                              w=w, paso=paso, seed=seed)
        dep = diagnostico_dependencia(v)
        filas.append({"serie": nombre, **d, "acf1": dep["acf1"]})
    return pd.DataFrame(filas).set_index("serie")
