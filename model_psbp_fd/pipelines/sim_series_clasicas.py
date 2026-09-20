"""
sim_series_clasicas.py
======================
Mecanismo COMUN de los "metodos basados en series de tiempo clasicas" del anexo
(`docs/01 Anexo.tex`, seccion `ane_00_01_metodos_series_tiempo`): los
Algoritmos A-1, A-2 y A-3 simulan una trayectoria escalar Z_1, ..., Z_{(B+T)L}
y la cortan en bloques contiguos de largo L; la curva t es el bloque t leido
sobre la grilla {tau_l}. Los tres difieren SOLO en el proceso escalar. Este
modulo concentra todo lo que no es el proceso escalar:

* el corte en `T` bloques y el descarte de `burn_in` bloques iniciales;
* el bloque anterior (`prev`) que da predictor a cada origen, incluido el
  primero cuando `burn_in > 0`;
* la funcion media aditiva, el ruido de observacion y las semillas por replica;
* el ensamblado de `SalidaSimulacion` y el esqueleto del diagnostico.

Cada algoritmo aporta un `simulador(rng, n_total)` que devuelve la trayectoria
`Z` (n_total,) y un dict con las series latentes que quiera exponer
(`{"regimen": S, "h": h}`); opcionalmente un hook `oraculos` que calcula las
medias condicionales de un rezago. Nada latente llega a la estimacion: todo va a
`SalidaSimulacion.internos`.

Contrato de reproducibilidad
----------------------------
Por replica se abre UN generador `default_rng(hija)` y se consume en este orden:
primero el simulador (la trayectoria completa), despues el ruido de observacion.
Es exactamente el orden que tenia `sim_escenario_A1` antes de factorizar, de
modo que la corrida 61 sigue dando las mismas curvas con la misma semilla; cambiar
el orden de consumo cambiaria las trayectorias en silencio.

Sobre `burn_in`
---------------
Segun el proceso puede ser cosmetico (A-1: Davies-Harte sale estacionario;
A-2: la cadena parte de su ley estacionaria y solo `u_0 = 0` es transitorio) o
funcional (A-3: `h_1 = sigma_a^2` y `Z_0 = 0` son condiciones iniciales). En
todos los casos el bloque descartado inmediatamente anterior al primero retenido
existe y es el que le da predictor al origen t = 1.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, toeplitz

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    aplicar_ruido_observacion,
    diagnostico_comun,
    evaluar_media,
    grilla_regular,
    semillas_replicas,
)

__all__ = [
    "acf_empirica_serie",
    "bloques_de_serie",
    "oraculo_lineal_un_rezago",
    "generar_serie_segmentada",
    "diagnostico_serie_escalar",
    "r2_empirico_media_condicional",
]

# simulador(rng, n_total) -> (Z (n_total,), series latentes {nombre: (n_total,)})
Simulador = Callable[[np.random.Generator, int], tuple]
# oraculos(prev (T, L), Z (n_total,), latentes, ctx) -> {clave_internos: (T, L)}
Oraculos = Callable[[np.ndarray, np.ndarray, dict, dict], dict]


# ==========================================================================
# SEGMENTACION
# ==========================================================================

def bloques_de_serie(Z: np.ndarray, L: int, burn: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Corta Z en bloques contiguos de largo L y devuelve (bloques, prev).

    Z : (n_total,) con n_total = (burn + T) L.
    bloques : (T, L), la serie ya sin los `burn` bloques iniciales.
    prev    : (T, L), el bloque inmediatamente anterior a cada uno. Con
        burn > 0 el anterior al primero retenido existe y se usa; sin
        calentamiento el primer origen no tiene predictor y va en cero.
    """
    n_bloques = Z.size // L
    if n_bloques * L != Z.size:
        raise ValueError(f"len(Z)={Z.size} no es multiplo de L={L}.")
    todos = Z.reshape(n_bloques, L)
    bloques = todos[burn:]
    prev = (todos[burn - 1:-1] if burn > 0
            else np.vstack([np.zeros((1, L)), bloques[:-1]]))
    return bloques, prev


def acf_empirica_serie(Z: np.ndarray, k: int) -> float:
    """Autocorrelacion muestral de Z al rezago k, promediada sobre replicas."""
    Z = np.atleast_2d(Z)
    vals = []
    for r in range(Z.shape[0]):
        z = Z[r] - Z[r].mean()
        den = float((z * z).sum())
        vals.append(float((z[:-k] * z[k:]).sum() / den) if den > 0 else np.nan)
    return float(np.nanmean(vals))


# ==========================================================================
# ORACULO LINEAL DE UN REZAGO (POR MOMENTOS)
# ==========================================================================

def oraculo_lineal_un_rezago(gamma: np.ndarray, L: int, jitter: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Predictor LINEAL optimo del bloque t dado el bloque t-1, y la covarianza de
    su residuo, a partir de las autocovarianzas estacionarias de la serie.

    Retorna (A, Sigma_resid) con A = Sigma_21 Sigma_11^{-1}, de forma (L, L).
    Solo usa segundos momentos: es la media condicional EXACTA cuando la serie
    es gaussiana (A-1) y el mejor predictor lineal --no la media condicional--
    cuando no lo es (A-2).
    """
    S22 = toeplitz(gamma[:L])                                  # (L, L)
    S11 = S22 + jitter * max(gamma[0], 1.0) * np.eye(L)
    idx = np.arange(L)
    S21 = gamma[np.abs(L + idx[:, None] - idx[None, :])]       # (L, L)
    c, low = cho_factor(S11, lower=True)
    A = cho_solve((c, low), S21.T).T                           # Sigma_21 S11^{-1}
    return A, S22 - A @ S21.T


# ==========================================================================
# GENERADOR COMUN
# ==========================================================================

def generar_serie_segmentada(
    cfg: ConfigObservacion,
    simulador: Simulador,
    *,
    oraculos: Optional[Oraculos] = None,
    ctx: Optional[dict] = None,
    internos_extra: Optional[dict] = None,
    resumen: Optional[Callable[[SalidaSimulacion], dict]] = None,
    diagnosticar: bool = True,
) -> SalidaSimulacion:
    """
    Genera R replicas: simula la trayectoria escalar, la corta en bloques y
    aplica el esquema de observacion comun.

    simulador : ver `Simulador`. Es lo unico que cambia entre algoritmos.
    oraculos : hook opcional que recibe el bloque anterior de cada origen y
        devuelve cantidades condicionales de un rezago, (T, L) por clave. Las
        claves "media_condicional*" vienen SIN la funcion media (se suma aqui) y
        "media_condicional" es obligatoria; las demas (p. ej. desviaciones) se
        guardan en `internos` tal cual.
    ctx : objetos comunes que el hook necesita (operadores, parametros).
    internos_extra : se copia tal cual a `internos` (cantidades globales).
    resumen : control de calidad del algoritmo, invocado con la salida ya armada.

    `internos` recibe "serie_escalar" (R, n_total), cada serie latente como
    (R, n_total) bajo su nombre, y lo que devuelva `oraculos`.
    """
    cfg.validar()
    L, T, R = int(cfg.L), int(cfg.T), int(cfg.R)
    burn = int(cfg.burn_in)
    n_total = (burn + T) * L
    ctx = ctx or {}

    tau = grilla_regular(L)
    media = evaluar_media(cfg.media_fn, tau)

    hijas, registro = semillas_replicas(cfg.seed, R)
    curvas = np.empty((R, T, L))
    observaciones = np.empty((R, T, L))
    series = np.empty((R, n_total))
    latentes_r: dict[str, np.ndarray] = {}
    oraculos_r: dict[str, np.ndarray] = {}

    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        Z, lat = simulador(rng, n_total)
        Z = np.asarray(Z, dtype=float)
        if Z.shape != (n_total,):
            raise ValueError(f"El simulador devolvio {Z.shape}; se esperaba "
                             f"({n_total},).")
        bloques, prev = bloques_de_serie(Z, L, burn)

        curvas[r] = bloques + media
        observaciones[r] = aplicar_ruido_observacion(curvas[r], cfg.sigma_obs, rng)
        series[r] = Z

        for nombre, v in lat.items():
            latentes_r.setdefault(nombre, np.empty((R, n_total)))[r] = v
        if oraculos is not None:
            for clave, m in oraculos(prev, Z, lat, ctx).items():
                # Solo las medias condicionales llevan la funcion media aditiva.
                if clave.startswith("media_condicional"):
                    m = m + media
                oraculos_r.setdefault(clave, np.empty((R, T, L)))[r] = m

    if oraculos is not None and "media_condicional" not in oraculos_r:
        raise ValueError("El hook `oraculos` debe devolver 'media_condicional'.")

    internos = {"serie_escalar": series, **latentes_r, **oraculos_r,
                **(internos_extra or {})}
    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=media,
        semillas=registro,
        config=cfg,
        internos=internos,
    )
    if diagnosticar and resumen is not None:
        salida.diagnostico = resumen(salida)
    return salida


# ==========================================================================
# ESQUELETO DEL DIAGNOSTICO
# ==========================================================================

def r2_empirico_media_condicional(salida: SalidaSimulacion,
                                  clave: str = "media_condicional") -> float:
    """
    R^2 = 1 - SSE/SST de la media condicional guardada en `internos[clave]`
    contra las curvas generadas, con la media funcional restada. Se omite el
    primer origen. Es la cifra que un metodo con n_lags = 1 puede aspirar a
    igualar, y no mas.
    """
    X = salida.curvas - salida.media
    Mc = salida.internos[clave] - salida.media
    sse = float(((X[:, 1:] - Mc[:, 1:]) ** 2).sum())
    sst = float((X[:, 1:] ** 2).sum())
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def diagnostico_serie_escalar(salida: SalidaSimulacion) -> dict:
    """
    Diagnostico comun a los tres algoritmos: el de `diagnostico_comun` mas la
    autocorrelacion EMPIRICA de la serie escalar en los rezagos que importan
    (1 dentro de la curva, L entre curvas consecutivas, 2L a dos curvas), su
    desviacion, el corte T0 de referencia y la finitud.
    """
    cfg = salida.config
    L, T = int(cfg.L), int(cfg.T)
    Z = salida.internos["serie_escalar"]
    prop = float(getattr(cfg, "prop_train_referencia", 0.70))
    return {
        **diagnostico_comun(salida),
        "rho_empirica_lag1": acf_empirica_serie(Z, 1),
        "rho_empirica_lagL": acf_empirica_serie(Z, L),
        "rho_empirica_lag2L": acf_empirica_serie(Z, 2 * L),
        "sd_Z_empirica": float(Z.std()),
        "T0_referencia": int(np.floor(prop * T)),
        "burn_in_bloques": int(cfg.burn_in),
        "todo_finito": bool(np.all(np.isfinite(salida.curvas))),
    }
