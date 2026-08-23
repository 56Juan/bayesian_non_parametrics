r"""
diagnostics_mcmc.py — Diagnostico de convergencia, separado del dibujo
======================================================================

Estas funciones vivian dentro de `graphics/viz_traces.py`, de modo que obtener
la tabla de diagnosticos exigia generar las figuras. Aqui son el calculo, y
`viz_traces` las importa en lugar de redefinirlas: la ACF, el ESS y R-hat
tienen una sola definicion en el proyecto, igual que la cuadratura y el
Cholesky.

Que se diagnostica y con que criterio
-------------------------------------
    ESS (Geyer 1992)   tamano de muestra efectivo. La regla de truncamiento por
                       pares positivos es la adecuada para cadenas reversibles
                       y no requiere elegir un lag maximo a mano.
    Geweke z           compara el primer 10% con el ultimo 50% de la cadena.
                       |z| > 2 indica que la cadena aun se desplaza.
    R-hat              Gelman-Rubin entre cadenas. Requiere al menos dos; con
                       una sola devuelve NaN en lugar de un numero enganoso.

Una advertencia propia de este modelo
-------------------------------------
El PSBPM es una mezcla con etiquetas intercambiables. R-hat y ESS sobre
parametros indexados por componente (beta_hj, tau_h, alpha_h de un h fijo) son
poco informativos: dos cadenas pueden describir la misma posterior con las
etiquetas permutadas y arrojar R-hat enorme sin que nada este mal. Por eso los
diagnosticos se calculan sobre cantidades INVARIANTES a la permutacion --el
promedio sobre componentes, que es lo que hace `_extraer_traza_variable` con
las trazas de tres dimensiones-- y sobre los parametros que no dependen de la
etiqueta: pi_j, w_j y el numero de atomos ocupados.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "autocorr",
    "ess_geyer",
    "geweke_z",
    "gelman_rubin",
    "extraer_traza_variable",
    "diagnostico_variable",
    "tabla_diagnosticos",
    "resumen_convergencia",
]

# Umbrales de lectura. Se declaran una vez para que la tabla, el veredicto y
# los graficos no puedan discrepar entre si.
UMBRAL_RHAT = 1.1
UMBRAL_ESS = 100.0
UMBRAL_GEWEKE = 2.0


# ==========================================================================
# ESTADISTICOS BASE
# ==========================================================================

def autocorr(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """ACF muestral hasta `max_lag` (lag 0 = 1). Implementacion por FFT."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return np.array([1.0])
    xc = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    if acf[0] == 0:
        return np.zeros(min(max_lag, n - 1) + 1)
    acf = acf / acf[0]
    return acf[: max_lag + 1]


def ess_geyer(x: np.ndarray) -> float:
    """
    Tamano de muestra efectivo con la regla de truncamiento de Geyer (1992):
    se acumulan pares consecutivos de autocorrelaciones y se corta en el primer
    par negativo, que es el estimador de varianza inicial positiva.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 4 or np.allclose(x, x[0]):
        return float(n)
    rho = autocorr(x, max_lag=min(n - 1, 1000))
    s = 0.0
    for k in range(1, len(rho) - 1, 2):
        par = rho[k] + rho[k + 1]
        if par < 0:
            break
        s += par
    tau_int = 1.0 + 2.0 * s
    return float(n / tau_int) if tau_int > 0 else float(n)


def geweke_z(x: np.ndarray, primero: float = 0.1, ultimo: float = 0.5) -> float:
    """Test z de Geweke entre el primer `primero` y el ultimo `ultimo` de la cadena."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    n_a, n_b = int(primero * n), int(ultimo * n)
    if n_a < 2 or n_b < 2:
        return float("nan")
    xa, xb = x[:n_a], x[-n_b:]
    var_a = xa.var(ddof=1) / max(ess_geyer(xa), 1.0)
    var_b = xb.var(ddof=1) / max(ess_geyer(xb), 1.0)
    den = np.sqrt(var_a + var_b)
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float((xa.mean() - xb.mean()) / den)


def gelman_rubin(cadenas: np.ndarray) -> float:
    """R-hat de Gelman-Rubin sobre una matriz (m_cadenas, n_iter)."""
    C = np.asarray(cadenas, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(f"cadenas debe ser 2D (m, n); recibido {C.shape}.")
    m, n = C.shape
    if m < 2 or n < 2:
        return float("nan")
    B = n * C.mean(axis=1).var(ddof=1)
    W = C.var(axis=1, ddof=1).mean()
    if W == 0:
        return float("nan")
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


# ==========================================================================
# EXTRACCION DESDE LAS TRAZAS DEL MUESTREADOR
# ==========================================================================

def extraer_traza_variable(traza: np.ndarray, j: int, burn: int) -> np.ndarray:
    """
    Serie post-calentamiento de la variable `j` a partir de una traza cruda.

    (nsim, p)     -> columna j tal cual (pi_j, w_j, osum_j).
    (nsim, N, p)  -> PROMEDIO sobre las N componentes de la mezcla. El promedio
                     es invariante a la permutacion de etiquetas, que en un
                     modelo de mezcla cambia entre cadenas sin que la posterior
                     cambie; usar una componente fija haria que R-hat midiera
                     el etiquetado y no la convergencia.
    """
    A = np.asarray(traza)
    if A.ndim == 3:
        return A[burn:, :, j].mean(axis=1).astype(np.float64)
    if A.ndim == 2:
        return A[burn:, j].astype(np.float64)
    raise ValueError(f"traza con ndim={A.ndim} no soportada.")


def diagnostico_variable(cadenas: np.ndarray) -> dict:
    """
    ESS, Geweke y R-hat sobre una matriz (m_cadenas, n_post) ya post-burn.

    `converge` es el veredicto conjunto contra los tres umbrales del modulo.
    """
    C = np.atleast_2d(np.asarray(cadenas, dtype=np.float64))
    ess = np.array([ess_geyer(C[c]) for c in range(C.shape[0])])
    gew = np.array([geweke_z(C[c]) for c in range(C.shape[0])])
    rhat = gelman_rubin(C)

    gew_max = float(np.nanmax(np.abs(gew))) if np.any(np.isfinite(gew)) else float("nan")
    ok_rhat = (not np.isfinite(rhat)) or rhat < UMBRAL_RHAT
    ok_gew = (not np.isfinite(gew_max)) or gew_max < UMBRAL_GEWEKE
    return {
        "n_cadenas":  int(C.shape[0]),
        "n_post":     int(C.shape[1]),
        "ess_min":    float(ess.min()),
        "ess_mean":   float(ess.mean()),
        "geweke_max": gew_max,
        "rhat":       rhat,
        "converge":   bool(ok_rhat and ok_gew and ess.min() > UMBRAL_ESS),
    }


def tabla_diagnosticos(models_chains: Dict, burn: int,
                       component_idx: Optional[Sequence[int]] = None,
                       claves: Sequence[str] = ("betajhout", "pijout"),
                       feature_names: Optional[Sequence[str]] = None
                       ) -> pd.DataFrame:
    """
    Tabla de diagnosticos para todas las componentes, variables y cadenas.

    models_chains : {k: {c: modelo}} donde cada modelo expone `.traces` (dict de
        arreglos crudos) y `.feature_names_`. Es la estructura que arman los
        notebooks de resultados a partir de los .mat.
    burn   : iteraciones descartadas.
    claves : trazas a diagnosticar. Por defecto los coeficientes de regresion
        (promediados sobre componentes) y las probabilidades de inclusion, que
        son las dos cantidades que se reportan en la tesis.

    Retorna una fila por (componente FPCA, parametro, variable). No dibuja nada:
    ese es el punto de este modulo.
    """
    _ALIAS = {"betajhout": "beta_j", "pijout": "p_j", "wjout": "w_j",
              "psijhout": "psi_j", "gammajhout": "gamma_j", "osumout": "osum_j"}

    filas = []
    for k in sorted(models_chains):
        cadenas_k = sorted(models_chains[k])
        if not cadenas_k:
            continue
        m0 = models_chains[k][cadenas_k[0]]
        nombres = list(feature_names) if feature_names is not None \
            else list(getattr(m0, "feature_names_", []))
        fpc = (component_idx[k] + 1) if component_idx is not None else k + 1

        for clave in claves:
            if clave not in m0.traces:
                continue
            p = m0.traces[clave].shape[-1]
            etiquetas = nombres if len(nombres) == p else [f"x{j+1}" for j in range(p)]
            for j in range(p):
                mat = np.vstack([
                    extraer_traza_variable(models_chains[k][c].traces[clave], j, burn)
                    for c in cadenas_k])
                filas.append({
                    "componente": fpc,
                    "param":      _ALIAS.get(clave, clave),
                    "variable":   etiquetas[j],
                    **diagnostico_variable(mat),
                })

    return pd.DataFrame(filas)


def resumen_convergencia(tabla: pd.DataFrame) -> dict:
    """
    Veredicto agregado sobre la tabla de `tabla_diagnosticos`.

    Se reporta el peor caso y no el promedio: una sola variable sin converger
    invalida la posterior conjunta, y promediar R-hat la esconde.
    """
    if tabla.empty:
        return {"n_variables": 0, "todo_converge": False}
    return {
        "n_variables":     int(len(tabla)),
        "rhat_max":        float(np.nanmax(tabla["rhat"])),
        "ess_min":         float(np.nanmin(tabla["ess_min"])),
        "geweke_max":      float(np.nanmax(tabla["geweke_max"])),
        "n_no_converge":   int((~tabla["converge"]).sum()),
        "variables_malas": tabla.loc[~tabla["converge"],
                                     ["componente", "param", "variable"]]
                                .to_dict("records"),
        "todo_converge":   bool(tabla["converge"].all()),
        "umbrales":        {"rhat": UMBRAL_RHAT, "ess": UMBRAL_ESS,
                            "geweke": UMBRAL_GEWEKE},
    }
