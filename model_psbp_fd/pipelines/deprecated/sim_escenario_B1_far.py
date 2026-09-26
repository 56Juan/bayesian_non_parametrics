"""
sim_escenario_B1_far.py
========================
HISTORICO. Generador original del Algoritmo B-1 (usado por las corridas 71-73,
que no se re-ejecutan). El anexo redefinio B-1 como una mezcla de mecanismos
(ver `pipelines/sim_escenario_B1.py`, vigente); este modulo ya NO corresponde
a lo que `docs/01 Anexo.tex` llama B-1. Se conserva solo para que 71_01, 72_01
y 73_01 sigan siendo legibles como documento historico; no se usa en corridas
nuevas y no se importa desde `pipelines` directamente.

Contenido original: proceso autorregresivo funcional de orden uno, lineal,
gaussiano y homogeneo, con nucleo autorregresivo EXPONENCIAL.

Modelo generador
----------------
    X_t(tau) = mu(tau) + int_0^1 psi(tau, s) (X_{t-1}(s) - mu(s)) ds + eps_t(tau)

    psi(tau, s) = c exp{ -|tau - s| / gamma }

    Cov(eps_t(tau), eps_t(s)) = sigma_eps^2 exp{ -(tau - s)^2 / (2 ell^2) }

Es el mismo mecanismo de `sim_escenario_1` con el nucleo cambiado: el
exponencial no es diferenciable en tau = s, de modo que el operador suaviza
menos y transmite a la curva siguiente mas detalle fino de la anterior. La
dinamica se simula con la misma recursion y consume el generador aleatorio en el
mismo orden, de modo que con `nucleo="exponencial"` y los mismos parametros
`sim_escenario_1` y este modulo dan las mismas trayectorias.

Papel en el estudio
-------------------
Es un control, como el A-1: la ley condicional es gaussiana, unimodal y
homogenea, y el FAR(1) esta correctamente especificado; el resultado esperado
es el empate.

Lo que aporta sobre `sim_escenario_1`
-------------------------------------
* `media_condicional` --el oraculo de un rezago-- y `hs_oraculo_L2` en el
  diagnostico. Aqui el oraculo es EXACTO y es el propio operador:

      E[X_t | X_{t-1}] = mu + Psi (X_{t-1} - mu),      ||Psi||_HS = hs_norm.

* El esqueleto reutilizable por B-2 y B-3 (`_ConfigBaseB`,
  `simular_far_centrado`, `generar_far_funcional`): ambos heredan la media, la
  innovacion y el esquema de observacion de este algoritmo y solo cambian el
  operador por paso, la escala de la innovacion y el desplazamiento aditivo.

Convenciones de tiempo
----------------------
El paso absoluto `i = 0, ..., burn_in + T - 1` corresponde a la curva
`t = i - burn_in + 1`; las retenidas son `t = 1, ..., T` y el calentamiento
ocupa `t <= 0`. `Y[0]` es el estado al cierre del calentamiento y es el
predictor del primer origen retenido, de modo que el oraculo existe tambien
para `t = 1`.

Contrato de reproducibilidad
----------------------------
Por replica se abre UN generador `default_rng(hija)` y se consume en este orden:
la trayectoria completa (`Y_0`, calentamiento y curvas retenidas), despues el
ruido de observacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from ..sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    aplicar_ruido_observacion,
    diagnostico_comun,
    evaluar_media,
    factor_cholesky,
    generador_innovacion,
    grilla_regular,
    matriz_covarianza_innovacion,
    matriz_operador_ar,
    norma_hilbert_schmidt,
    pesos_trapezoidales,
    semillas_replicas,
    NUCLEOS_AR,
)
from ..sim_series_clasicas import r2_empirico_media_condicional

__all__ = [
    "media_seno",
    "ConfigEscenarioB1",
    "generar_escenario_B1",
    "resumen_escenario_B1",
]


def media_seno(tau: np.ndarray) -> np.ndarray:
    """mu(tau) = sin(2 pi tau): la media de los tres algoritmos B (Cuadro tab:ane_algB1)."""
    return np.sin(2.0 * np.pi * np.asarray(tau, dtype=float))


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class _ConfigBaseB(ConfigObservacion):
    """
    Parametros comunes a B-1, B-2 y B-3: media, innovacion y nucleo.

    sigma_eps, ell : escala y longitud de correlacion de la innovacion.
    nucleo : forma de psi; "exponencial" en el anexo.
    prop_train_referencia : solo para el diagnostico; el corte T0 real lo fija
        el notebook.
    """

    media_fn: Optional[Callable[[np.ndarray], np.ndarray]] = media_seno
    sigma_eps: float = 1.0
    ell: float = 0.5
    nucleo: str = "exponencial"
    prop_train_referencia: float = 0.70

    def validar(self) -> None:
        super().validar()
        if self.sigma_eps <= 0:
            raise ValueError("sigma_eps debe ser positivo.")
        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")
        if self.nucleo not in NUCLEOS_AR:
            raise ValueError(f"nucleo debe ser uno de {NUCLEOS_AR}.")
        if not 0.0 < self.prop_train_referencia < 1.0:
            raise ValueError("prop_train_referencia debe estar en (0, 1).")


@dataclass
class ConfigEscenarioB1(_ConfigBaseB):
    """
    Algoritmo B-1. Valores del anexo (Cuadro tab:ane_algB1): mu = sin(2 pi tau),
    gamma = 0.3, ||Psi||_HS = 0.7, sigma_eps = 1.0, ell = 0.5.

    gamma   : alcance del nucleo autorregresivo.
    hs_norm : norma HS objetivo del operador; en (0, 1) para que exista
              solucion estacionaria.
    """

    gamma: float = 0.3
    hs_norm: float = 0.7

    def validar(self) -> None:
        super().validar()
        if self.gamma <= 0:
            raise ValueError("gamma debe ser positivo.")
        if not 0.0 < self.hs_norm < 1.0:
            raise ValueError(f"hs_norm={self.hs_norm}: debe estar en (0, 1).")


# ==========================================================================
# RECURSION
# ==========================================================================

def simular_far_centrado(
    Psi0: np.ndarray,
    Psi1: Optional[np.ndarray],
    w_pasos: np.ndarray,
    kappa_pasos: np.ndarray,
    chol_K: np.ndarray,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Itera Y_i = Psi_i Y_{i-1} + kappa_i eps_i sobre el estado centrado y
    devuelve Y de forma (T + 1, L), con T = len(w_pasos) - burn_in.

    Psi_i = (1 - w_i) Psi0 + w_i Psi1; con `Psi1 = None` el operador es
    constante (B-1 y B-3). Y[0] es el estado al cierre del calentamiento; Y[t],
    t = 1..T, la curva centrada retenida t.

    Con `w = 0` y `kappa = 1` es exactamente la recursion de
    `sim_escenario_1.simular_trayectoria_far1` (misma inicializacion y mismo
    orden de extraccion de la innovacion).
    """
    n_pasos = int(w_pasos.size)
    L = Psi0.shape[0]
    if kappa_pasos.size != n_pasos:
        raise ValueError("w_pasos y kappa_pasos deben tener el mismo largo.")
    innovacion = generador_innovacion(chol_K, rng)

    Y = innovacion()
    estados = np.empty((n_pasos - burn_in + 1, L))
    for i in range(n_pasos):
        if i == burn_in:
            estados[0] = Y
        if Psi1 is None:
            avance = Psi0 @ Y
        else:
            avance = (1.0 - w_pasos[i]) * (Psi0 @ Y) + w_pasos[i] * (Psi1 @ Y)
        Y = avance + kappa_pasos[i] * innovacion()
        if i >= burn_in:
            estados[i - burn_in + 1] = Y
    if burn_in == n_pasos:
        estados[0] = Y
    return estados


def media_condicional_lineal(
    Y_prev: np.ndarray,
    Psi0: np.ndarray,
    Psi1: Optional[np.ndarray],
    w_ret: np.ndarray,
) -> np.ndarray:
    """
    Psi_t Y_{t-1} para t = 1..T, vectorizado: Y_prev (T, L), w_ret (T,).
    Es (1 - w) Y Psi0' + w Y Psi1' porque el operador es lineal en w.
    """
    if Psi1 is None:
        return Y_prev @ Psi0.T
    return ((1.0 - w_ret)[:, None] * (Y_prev @ Psi0.T)
            + w_ret[:, None] * (Y_prev @ Psi1.T))


def generar_far_funcional(
    cfg: _ConfigBaseB,
    Psi0: np.ndarray,
    Psi1: Optional[np.ndarray],
    w_pasos: np.ndarray,
    kappa_pasos: np.ndarray,
    desplazamiento_ret: Optional[np.ndarray] = None,
    internos_extra: Optional[dict] = None,
    resumen: Optional[Callable[[SalidaSimulacion], dict]] = None,
    diagnosticar: bool = True,
) -> SalidaSimulacion:
    """
    Esqueleto comun de B-1, B-2 y B-3: simula, suma media y desplazamiento,
    calcula el oraculo de un rezago y aplica el esquema de observacion.

    desplazamiento_ret : (T + 1, L) o None; termino aditivo determinista para
        t = 0..T (episodio de B-3). Fila 0 es t = 0 (cierre del calentamiento).

    El oraculo condiciona en X_{t-1} y en el calendario determinista:

        E[X_t | X_{t-1}] = mu + d_t + Psi_t (X_{t-1} - mu - d_{t-1}).

    `internos` recibe `media_condicional` y `media_condicional_lineal_sin_calendario`
    (mu + Psi_t (X_{t-1} - mu): lo que hace el operador verdadero sin conocer d_t),
    ambas (R, T, L).
    """
    cfg.validar()
    R, T, L = int(cfg.R), int(cfg.T), int(cfg.L)
    burn = int(cfg.burn_in)
    if w_pasos.size != burn + T:
        raise ValueError(f"w_pasos debe tener burn_in + T = {burn + T} entradas.")

    tau = grilla_regular(L)
    mu = evaluar_media(cfg.media_fn, tau)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)
    w_ret = w_pasos[burn:]                                 # (T,) t = 1..T
    d = (np.zeros((T + 1, L)) if desplazamiento_ret is None
         else np.asarray(desplazamiento_ret, dtype=float))

    hijas, registro = semillas_replicas(cfg.seed, R)
    curvas = np.empty((R, T, L))
    observaciones = np.empty((R, T, L))
    mc = np.empty((R, T, L))
    mc_sin = np.empty((R, T, L))

    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        Y = simular_far_centrado(Psi0, Psi1, w_pasos, kappa_pasos, chol_K,
                                 burn, rng)
        curvas[r] = mu + d[1:] + Y[1:]
        observaciones[r] = aplicar_ruido_observacion(curvas[r], cfg.sigma_obs, rng)
        avance = media_condicional_lineal(Y[:-1], Psi0, Psi1, w_ret)
        mc[r] = mu + d[1:] + avance
        # Sin calendario: el operador actua sobre la curva observada centrada
        # tal cual, X_{t-1} - mu = Y_{t-1} + d_{t-1}.
        avance_sin = media_condicional_lineal(Y[:-1] + d[:-1], Psi0, Psi1, w_ret)
        mc_sin[r] = mu + avance_sin

    internos = {
        "operador": Psi0,
        "operador_final": Psi1 if Psi1 is not None else Psi0,
        "cov_innovacion": K,
        "pesos_cuadratura": pesos_trapezoidales(tau),
        "peso_operador_final": np.asarray(w_ret),
        "escala_innovacion": np.asarray(kappa_pasos[burn:]),
        "media_condicional": mc,
        "media_condicional_lineal_sin_calendario": mc_sin,
        **(internos_extra or {}),
    }
    salida = SalidaSimulacion(
        observaciones=observaciones, curvas=curvas, grilla=tau, media=mu,
        semillas=registro, config=cfg, internos=internos,
    )
    if diagnosticar and resumen is not None:
        salida.diagnostico = resumen(salida)
    return salida


# ==========================================================================
# GENERADOR B-1
# ==========================================================================

def generar_escenario_B1(cfg: ConfigEscenarioB1,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """
    Genera R replicas del Algoritmo B-1: FAR(1) lineal gaussiano homogeneo con
    nucleo exponencial. El operador y la factorizacion de la innovacion se
    construyen una sola vez y se comparten entre replicas.
    """
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm, nucleo=cfg.nucleo)
    n = int(cfg.burn_in) + int(cfg.T)
    return generar_far_funcional(
        cfg, Psi, None, np.zeros(n), np.ones(n),
        resumen=resumen_escenario_B1, diagnosticar=diagnosticar,
    )


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def r2_teorico_far_estacionario(Psi: np.ndarray, K: np.ndarray) -> float:
    """
    R^2 = 1 - tr(K) / tr(Sigma) del oraculo Psi Y_{t-1}, con Sigma la covarianza
    estacionaria de Y: Sigma = Psi Sigma Psi' + K (ecuacion de Lyapunov). Suma
    sobre los puntos de la grilla, igual que `r2_empirico_media_condicional`.
    """
    Sigma = solve_discrete_lyapunov(Psi, K)
    return float(1.0 - np.trace(K) / np.trace(Sigma))


def resumen_escenario_B1(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del B-1: el diagnostico comun, la norma HS efectiva del
    operador contra la objetivo, el radio espectral y el R^2 del oraculo de un
    rezago (teorico por Lyapunov y medido sobre las curvas).

    `hs_oraculo_L2` es ||Psi||_HS: el oraculo es el propio operador, de modo que
    el `_05` contrasta contra ella la norma del operador que estima el FAR.
    """
    if not isinstance(salida.config, ConfigEscenarioB1):
        raise TypeError("resumen_escenario_B1 requiere ConfigEscenarioB1; se "
                        f"recibio {type(salida.config).__name__}.")
    cfg = salida.config
    Psi = salida.internos["operador"]
    w_quad = salida.internos["pesos_cuadratura"]
    K = salida.internos["cov_innovacion"]

    hs = norma_hilbert_schmidt(Psi, w_quad)
    return {
        **diagnostico_comun(salida),
        "nucleo": cfg.nucleo,
        "hs_norm_objetivo": float(cfg.hs_norm),
        "hs_norm_efectiva": hs,
        "hs_norm_error_absoluto": abs(hs - cfg.hs_norm),
        "radio_espectral_operador": float(np.max(np.abs(np.linalg.eigvals(Psi)))),
        "estacionariedad_garantizada": bool(hs < 1.0),
        "hs_oraculo_L2": hs,
        "r2_oraculo_1rezago": r2_teorico_far_estacionario(Psi, K),
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "T0_referencia": int(np.floor(cfg.prop_train_referencia * cfg.T)),
    }
