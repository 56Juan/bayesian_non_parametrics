r"""
inclusion.py — Probabilidades posteriores de inclusion como cantidad reportable
===============================================================================

Tercer eje del diseno de simulacion (`docs/03 Modelo.tex §03_06`): contrastar
que variables selecciona el modelo contra la estructura de dependencia que el
generador realmente uso, conocida en cada escenario.

La materia prima ya estaba en disco --`psbp_train.m` guarda `gammajhout`
(nsim, N, p) y `osumout` (nsim, p)-- y existian graficos que la dibujaban, pero
no una funcion que devolviera la matriz lista para compararla con la verdad.
Eso es lo que hay aqui.

Dos definiciones de "inclusion", y no son la misma
--------------------------------------------------
    PIP GLOBAL      P(la variable j entra en ALGUNA componente ocupada).
                    Es el complemento de `osumout`, que el muestreador calcula
                    como 1{sum_h gamma_hj = 0} restringido a las componentes
                    efectivamente ocupadas (h <= max(S_i)). Responde "el modelo
                    usa esta variable". Es la cifra que corresponde contrastar
                    con la estructura del generador.

    PIP PROMEDIO    media sobre h de P(gamma_hj = 1). Responde "en que fraccion
    POR COMPONENTE  de los atomos aparece esta variable". Es sistematicamente
                    menor que la global y no debe confundirse con ella: bajo el
                    truncamiento en N atomos, la mayoria de los atomos casi
                    nunca esta ocupada y arrastra la media hacia el prior.

Cuando el generador es lineal y homogeneo --Escenario 1-- ambas coinciden mas o
menos; cuando hay cambio de regimen --Escenario 3-- se separan, y esa separacion
es en si misma el resultado interesante: la variable entra solo en los atomos
asociados a un regimen.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "pip_global",
    "pip_por_componente",
    "matriz_pip",
    "contraste_con_verdad",
]


# ==========================================================================
# EXTRACCION POR CADENA
# ==========================================================================

def pip_global(traces: Dict[str, np.ndarray], burn: int) -> np.ndarray:
    """
    P(variable j incluida en alguna componente ocupada), forma (p,).

    Complemento de `osumout`, que ya viene calculado por el muestreador sobre
    las componentes ocupadas. Si la traza no esta disponible se reconstruye
    desde `gammajhout`, aunque sin la restriccion a atomos ocupados: esa
    version cuenta atomos vacios y sobreestima, por lo que se prefiere
    `osumout` cuando existe.
    """
    if "osumout" in traces:
        osum = np.asarray(traces["osumout"], dtype=float)[burn:]
        return 1.0 - osum.mean(axis=0)
    if "gammajhout" not in traces:
        raise KeyError("Se requiere 'osumout' o 'gammajhout' en las trazas.")
    g = np.asarray(traces["gammajhout"], dtype=float)[burn:]     # (T, N, p)
    return (g.max(axis=1) > 0).mean(axis=0)


def pip_por_componente(traces: Dict[str, np.ndarray], burn: int) -> np.ndarray:
    """
    P(gamma_hj = 1) para cada atomo h y variable j, forma (N, p).

    Sin restringir a atomos ocupados: los atomos vacios muestrean del prior y
    su presencia en la matriz es informativa --muestra donde termina la senal
    y empieza el prior-- siempre que se lea como tal.
    """
    if "gammajhout" not in traces:
        raise KeyError("Se requiere 'gammajhout' en las trazas.")
    g = np.asarray(traces["gammajhout"], dtype=float)[burn:]     # (T, N, p)
    return g.mean(axis=0)


# ==========================================================================
# AGREGACION SOBRE CADENAS Y COMPONENTES FPCA
# ==========================================================================

def matriz_pip(models_chains: Dict, burn: int,
               component_idx: Optional[Sequence[int]] = None,
               feature_names: Optional[Sequence[str]] = None,
               incluir_dispersion: bool = True) -> pd.DataFrame:
    """
    Matriz PIP lista para reportar: filas = variables, columnas = componentes FPCA.

    models_chains : {k: {c: modelo}} con `.traces` y `.feature_names_`.
    incluir_dispersion : agrega columnas `<comp>_sd` con la desviacion entre
        cadenas. Con pocas cadenas la PIP de una sola no es estable, y publicar
        la media sin su dispersion oculta justamente el caso en que las cadenas
        discrepan sobre que variables importan --que es un fallo de mezcla
        disfrazado de resultado de seleccion.

    Las cadenas se promedian con peso igual, que es la mezcla que corresponde
    cuando todas tienen la misma longitud post-calentamiento.
    """
    columnas: Dict[str, pd.Series] = {}
    etiquetas: Optional[List[str]] = None

    for k in sorted(models_chains):
        cadenas_k = sorted(models_chains[k])
        if not cadenas_k:
            continue
        m0 = models_chains[k][cadenas_k[0]]
        nombres = list(feature_names) if feature_names is not None \
            else list(getattr(m0, "feature_names_", []))
        pips = np.column_stack([
            pip_global(models_chains[k][c].traces, burn) for c in cadenas_k])
        if not nombres or len(nombres) != pips.shape[0]:
            nombres = [f"x{j+1}" for j in range(pips.shape[0])]
        etiquetas = nombres

        fpc = (component_idx[k] + 1) if component_idx is not None else k + 1
        col = f"FPC {fpc}"
        columnas[col] = pd.Series(pips.mean(axis=1), index=nombres)
        if incluir_dispersion:
            sd = pips.std(axis=1, ddof=1) if pips.shape[1] > 1 \
                else np.zeros(pips.shape[0])
            columnas[f"{col}_sd"] = pd.Series(sd, index=nombres)

    if not columnas:
        return pd.DataFrame()
    tabla = pd.DataFrame(columnas)
    tabla.index.name = "variable"
    return tabla


# ==========================================================================
# CONTRASTE CONTRA LA ESTRUCTURA DEL GENERADOR
# ==========================================================================

def contraste_con_verdad(pip: pd.DataFrame, verdad: Dict[str, Sequence[str]],
                         umbral: float = 0.5) -> pd.DataFrame:
    """
    Cruza la matriz PIP con las variables que el generador realmente uso.

    pip    : salida de `matriz_pip` (se ignoran las columnas `_sd`).
    verdad : {"FPC 1": ["fpc_1_lag1", ...], ...} variables activas por
             componente segun el generador. En el Escenario 1 el operador es
             lineal y actua sobre toda la curva, de modo que en principio todas
             las FPC rezagadas son activas; en el 5 solo la componente
             subordinada lo es.
    umbral : PIP por encima del cual se declara seleccionada. 0.5 es la regla
             de la mediana del modelo, que bajo prior de inclusion simetrico es
             la decision de Bayes con perdida 0-1.

    Devuelve por componente: verdaderos/falsos positivos y negativos, tasa de
    seleccion correcta y el area bajo la curva ROC, que resume el ordenamiento
    sin depender del umbral elegido.
    """
    cols = [c for c in pip.columns if not c.endswith("_sd")]
    filas = []
    for col in cols:
        if col not in verdad:
            continue
        activas = set(verdad[col])
        p = pip[col]
        y = np.array([v in activas for v in p.index], dtype=bool)
        sel = (p.to_numpy() >= umbral)

        vp = int(np.sum(sel & y));  fp = int(np.sum(sel & ~y))
        fn = int(np.sum(~sel & y)); vn = int(np.sum(~sel & ~y))

        # AUC por el estadistico de Mann-Whitney sobre los rangos: fraccion de
        # pares (activa, inactiva) correctamente ordenados. Con una sola clase
        # presente no esta definida.
        if y.any() and (~y).any():
            r = pd.Series(p.to_numpy()).rank().to_numpy()
            n1, n0 = int(y.sum()), int((~y).sum())
            auc = float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        else:
            auc = float("nan")

        filas.append({
            "componente": col,
            "n_activas":  int(y.sum()),
            "VP": vp, "FP": fp, "FN": fn, "VN": vn,
            "sensibilidad": vp / max(vp + fn, 1),
            "especificidad": vn / max(vn + fp, 1),
            "exactitud": (vp + vn) / max(len(y), 1),
            "auc": auc,
            "pip_media_activas":   float(p[y].mean())  if y.any()    else float("nan"),
            "pip_media_inactivas": float(p[~y].mean()) if (~y).any() else float("nan"),
        })
    return pd.DataFrame(filas).set_index("componente") if filas else pd.DataFrame()
