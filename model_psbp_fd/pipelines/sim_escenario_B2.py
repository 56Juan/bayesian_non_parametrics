"""
sim_escenario_B2.py
====================
Algoritmo B-2 del anexo (`docs/01 Anexo.tex`, `ane_00_02_02_alg_b2`): la misma
mezcla de K = 3 mecanismos de B-1 (misma asignacion Z_t independiente, pi_k =
1/3, mismos rasgos w_j y medias mu_k), pero con mecanismos NO LINEALES.

Modelo generador (ecuaciones eq:ane_algB2 y siguientes)
----------------------------------------------------------
    X_t(tau) = mu_{Z_t}(tau) + f_{Z_t}(R_{t-1,1}, R_{t-1,2}, R_{t-1,3})(tau) + eps_t(tau)

con los tres mecanismos EXPLICITOS del anexo:

    f_1(R1,R2,R3)(tau) = beta_11(tau) R1 + beta_12(tau) R2 + beta_13(tau) R3
    f_2(R1,R2,R3)(tau) = beta_21(tau) R1^2 + beta_22(tau) R2 + beta_23(tau) R2 R3
    f_3(R1,R2,R3)(tau) = beta_31(tau) R1 R2 + beta_32(tau) R2^2 + beta_33(tau) R3^2

Igual que en B-1, se elige `beta_{kj}(tau) = b_{kj} phi_j(tau)` (mismo indice j
en el coeficiente y en el modo phi_j de `funciones_rasgo`), de modo que cada
mecanismo se reduce a un vector de 3 coeficientes escalares `B_MECANISMOS_B2[k]`
aplicado a un vector de rasgos TRANSFORMADO `z_k(R)` (identidad para k = 1,
[R1^2, R2, R2 R3] para k = 2, [R1 R2, R2^2, R3^2] para k = 3): ver
`transformar_rasgos_B2`.

A diferencia de B-1, la transformacion no lineal de k = 2 y k = 3 rompe el
desacople exacto por rasgo que tiene B-1 (un termino R1^2 alimenta el modo
phi_1 en el paso siguiente aunque provenga del CUADRADO de R1, no de R1
mismo), de modo que la estacionariedad ya no se sigue de una condicion
cerrada. Se calibro por simulacion directa del proceso de rasgos de 3
dimensiones (4000 pasos, 50 trayectorias, mismos B_MECANISMOS_B2 y
`sigma_eps`/`ell` heredados de B-1): |R| se mantiene acotado (< 2.8) y la
varianza por bloques de 500 pasos es estable.

El mecanismo 1 es identico en FORMA al mecanismo lineal de B-1 (misma
`transformar_identidad`) y usa SUS MISMOS coeficientes (`B_MECANISMOS_B2[0]`
= `B_MECANISMOS_B1[0]`); los de los mecanismos 2 y 3 (`B_MECANISMOS_B2[1:]`)
son mas grandes de lo que su contribucion a la varianza pueda sugerir a
primera vista: con R de escala moderada (Var(R) ~ 0.1-0.2), un termino
cuadratico R^2 tiene Var(R^2) ~ 2 Var(R)^2, intrinsecamente chico frente al
termino lineal de B-1. r2_oraculo_1rezago de B-2 (~0.13, T = 1000, R = 5,
sigma_obs = 0.25) queda por debajo del de B-1 (~0.50) por esta razon
estructural, no por falta de calibracion: subir aun mas los coeficientes
cuadraticos para igualar el R^2 de B-1 arriesgaria la estacionariedad (un
termino que crece con el CUADRADO de R retroalimenta mas fuerte cuanto mas
grande es R). Es el mismo patron que ya usa el Algoritmo C-1 del anexo
--dependencia lineal debil y dependencia cuadratica mas fuerte que un modelo
lineal no puede aprovechar--: el objetivo no es maximizar el techo agregado
sino dejar una señal no lineal genuina (ver docstring de `sim_escenario_B1`
para la comparacion completa con B-1 y B-3).

El oraculo de un rezago
------------------------
Igual que en B-1: Z_t es independiente de X_{t-1}, de modo que

    E[X_t | X_{t-1}] = sum_k pi_k (mu_k(tau) + f_k(R_{t-1})(tau))

es exacto y se calcula en el motor comun `simular_mezcla_curvas` (heredado de
`sim_escenario_B1`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim_comun import (
    SalidaSimulacion,
    aplicar_ruido_observacion,
    diagnostico_comun,
    factor_cholesky,
    grilla_regular,
    matriz_covarianza_innovacion,
    pesos_trapezoidales,
    semillas_replicas,
)
from .sim_escenario_B1 import (
    ConfigEscenarioB1,
    _pi_fija,
    funciones_rasgo,
    medias_mezcla,
    simular_mezcla_curvas,
)
from .sim_series_clasicas import r2_empirico_media_condicional

__all__ = [
    "transformar_rasgos_B2",
    "B_MECANISMOS_B2",
    "ConfigEscenarioB2",
    "generar_escenario_B2",
    "resumen_escenario_B2",
]


# ==========================================================================
# MECANISMOS B-2 (NO LINEALES)
# ==========================================================================

def transformar_rasgos_B2(R: np.ndarray, k: int) -> np.ndarray:
    """
    z_k(R) segun el mecanismo k (Cuadro tab:ane_algB2):

        k = 0 (f_1, lineal)       : [R1, R2, R3]
        k = 1 (f_2, cuadratico)   : [R1^2, R2, R2 R3]
        k = 2 (f_3, interacciones): [R1 R2, R2^2, R3^2]
    """
    R1, R2, R3 = R
    if k == 0:
        return np.array([R1, R2, R3])
    if k == 1:
        return np.array([R1 ** 2, R2, R2 * R3])
    return np.array([R1 * R2, R2 ** 2, R3 ** 2])


B_MECANISMOS_B2 = np.array([
    [0.85, 0.40, 0.15],
    [0.25, 0.70, 0.30],
    [0.30, 0.30, 0.25],
])
"""
Matriz (mecanismo x componente de z_k(R)) de los coeficientes escalares que
multiplican `phi_j(tau)` en cada `f_k`. f_1 usa los mismos coeficientes que el
mecanismo 1 de B-1; los de f_2/f_3 se mantienen moderados por la razon de
acotamiento explicada en el docstring del modulo.
"""


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioB2(ConfigEscenarioB1):
    """
    Algoritmo B-2: mismos parametros que B-1 (pi, sigma_eps, ell,
    prop_train_referencia); solo cambian los mecanismos (no lineales, ver
    `transformar_rasgos_B2` y `B_MECANISMOS_B2`).
    """


# ==========================================================================
# GENERADOR B-2
# ==========================================================================

def generar_escenario_B2(cfg: ConfigEscenarioB2,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """
    Genera R replicas del Algoritmo B-2: mezcla de 3 mecanismos NO lineales
    (uno lineal, uno cuadratico, uno con interacciones) con asignacion
    independiente de la curva anterior.
    """
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    pesos = pesos_trapezoidales(tau)
    Phi = funciones_rasgo(tau)
    MU = medias_mezcla(tau)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)
    pi = np.asarray(cfg.pi, dtype=float)
    pi_efectiva = _pi_fija(pi)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)
    R_, T, L = int(cfg.R), int(cfg.T), int(cfg.L)
    observaciones = np.empty((R_, T, L))
    curvas = np.empty((R_, T, L))
    mc = np.empty((R_, T, L))
    Z_all = np.empty((R_, T), dtype=int)
    Rasgos_all = np.empty((R_, T, 3))

    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        curvas_r, Z_r, R_r, mc_r = simular_mezcla_curvas(
            cfg, Phi, pesos, MU, B_MECANISMOS_B2, transformar_rasgos_B2,
            pi_efectiva, chol_K, int(cfg.burn_in), rng,
        )
        curvas[r] = curvas_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)
        mc[r] = mc_r
        Z_all[r] = Z_r
        Rasgos_all[r] = R_r

    media = pi @ MU
    internos = {
        "funciones_rasgo": Phi,
        "pesos_cuadratura": pesos,
        "medias_mecanismos": MU,
        "matriz_coeficientes": B_MECANISMOS_B2,
        "cov_innovacion": K,
        "pi": pi,
        "mecanismo": Z_all,
        "rasgos": Rasgos_all,
        "media_condicional": mc,
    }
    salida = SalidaSimulacion(
        observaciones=observaciones, curvas=curvas, grilla=tau, media=media,
        semillas=registro, config=cfg, internos=internos,
    )
    if diagnosticar:
        salida.diagnostico = resumen_escenario_B2(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def resumen_escenario_B2(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del B-2: igual al de B-1 (frecuencia de mecanismos,
    acotamiento de los rasgos, R^2 del oraculo de mezcla), mas la fraccion de
    la varianza de cada rasgo que aporta cada mecanismo, util para verificar
    que los terminos no lineales (k = 1, 2) no dominan por accidente.
    """
    if not isinstance(salida.config, ConfigEscenarioB2):
        raise TypeError("resumen_escenario_B2 requiere ConfigEscenarioB2; se "
                        f"recibio {type(salida.config).__name__}.")
    cfg = salida.config
    Z = salida.internos["mecanismo"]
    Rasgos = salida.internos["rasgos"]
    pi = salida.internos["pi"]
    frac = np.array([float((Z == k).mean()) for k in range(3)])
    var_por_mecanismo = np.array([
        Rasgos[Z == k].var(axis=0) if np.any(Z == k) else np.full(3, np.nan)
        for k in range(3)
    ])

    return {
        **diagnostico_comun(salida),
        "pi_objetivo": pi.tolist(),
        "pi_empirica": frac.tolist(),
        "pi_error_absoluto_max": float(np.max(np.abs(frac - pi))),
        "max_abs_rasgos": float(np.max(np.abs(Rasgos))),
        "var_rasgos": Rasgos.reshape(-1, 3).var(axis=0).tolist(),
        "var_rasgos_por_mecanismo": var_por_mecanismo.tolist(),
        "sigma_eps": float(cfg.sigma_eps),
        "ell": float(cfg.ell),
        "r2_oraculo_1rezago": r2_empirico_media_condicional(salida),
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "T0_referencia": int(np.floor(cfg.prop_train_referencia * cfg.T)),
    }
