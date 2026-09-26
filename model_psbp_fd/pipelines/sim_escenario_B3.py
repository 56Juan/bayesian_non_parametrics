"""
sim_escenario_B3.py
====================
Algoritmo B-3 del anexo (`docs/01 Anexo.tex`, `ane_00_02_03_alg_b3`): los
MISMOS tres mecanismos no lineales de B-2 (`transformar_rasgos_B2`,
`B_MECANISMOS_B2`, mismas medias mu_k y rasgos w_j = phi_j), pero con la
asignacion Z_t DEPENDIENTE de la curva anterior.

Modelo generador (eq:ane_algB3_probs y eq:ane_algB3)
-------------------------------------------------------
    P(Z_t = k | X_{t-1}) = softmax_k( q_k(R_{t-1,1}, R_{t-1,2}, R_{t-1,3}) ),
    X_t(tau) = mu_{Z_t}(tau) + f_{Z_t}(R_{t-1,1}, R_{t-1,2}, R_{t-1,3})(tau) + eps_t(tau).

Las funciones de asignacion q_k
---------------------------------
El anexo deja q_k "prefijada". Se eligen LINEALES en los tres rasgos (softmax
= regresion logistica multinomial, la forma estandar y mas simple que hace que
"la informacion disponible en X_{t-1} determina probabilisticamente que
mecanismo genera X_t", como pide el anexo), con el mecanismo 1 como referencia
(q_1 = 0, la normalizacion habitual del softmax):

    q_1(R) = 0
    q_2(R) = 1.0 R1 + 0.8 R2
    q_3(R) = -1.0 R1 + 0.8 R3

de modo que el mecanismo 2 (cuadratico) se favorece cuando la curva anterior
tiene nivel y pendiente altos, y el mecanismo 3 (interacciones) cuando tiene
nivel bajo o curvatura alta. Con R tipicamente < 2.1 (ver docstring de
`sim_escenario_B2`), los `q_k` tipicos son del orden de 1-2, lo que mueve el
softmax de forma apreciable respecto de (1/3, 1/3, 1/3): NO es una dependencia
cosmetica (medido con T = 1000, R = 5: `pi_empirica` se aleja de (1/3, 1/3,
1/3) hacia (0.36, 0.22, 0.42) y `prob_mecanismo_desviacion_del_centro` ~0.09,
ver docstring de `resumen_escenario_B3`). Verificado junto con B-1/B-2 en la
misma simulacion de calibracion del proceso de rasgos: con esta asignacion
dependiente, |R| se mantiene acotado (< 2.2) igual que con pi fija, y la
varianza por bloques es estable. `r2_oraculo_1rezago` de B-3 (~0.08, mismos
T, R, sigma_obs) es del mismo orden que el de B-2 (~0.13): la asignacion
dependiente recupera algo de senal sobre B-2 mediante el propio sorteo de
Z_t (ahora parcialmente predecible), pero el techo agregado sigue acotado por
la misma razon estructural de B-2 --mecanismos 2 y 3 cuadraticos con R de
escala moderada-- (ver docstring de `sim_escenario_B1` para la comparacion
completa entre los tres algoritmos).

Contraste con B-1/B-2
------------------------
En B-1 y B-2, Z_t es independiente de X_{t-1} y el oraculo de un rezago es la
mezcla con pesos FIJOS pi_k. Aqui los pesos de la mezcla dependen de R_{t-1},
que SI se observa (es una funcional de la curva anterior, no del futuro), de
modo que el oraculo de Bayes de un rezago sigue siendo exacto y en forma
cerrada:

    E[X_t | X_{t-1}] = sum_k p_k(R_{t-1}) (mu_k(tau) + f_k(R_{t-1})(tau)),
    p_k(R_{t-1}) = softmax_k(q_k(R_{t-1})),

calculado por el mismo motor `simular_mezcla_curvas` de `sim_escenario_B1`, que
ya acepta un `pi_efectiva` dependiente del rasgo. A diferencia del B-3
HISTORICO (cambio estructural con calendario determinista, ver
`pipelines/deprecated/sim_escenario_B3_cambio_estructural.py`), aqui NINGUN
calendario oculto separa al oraculo de lo que un modelo que solo observa
X_{t-1} puede en principio alcanzar: la brecha entre el PSBPM-FD y el techo es
puramente de especificacion (¿el modelo recupera el gating correcto?), no de
informacion oculta.

`internos["probabilidades_mecanismo"]` guarda p_k(R_{t-1}) para cada curva
retenida (ademas de `internos["mecanismo"]`, el Z_t efectivamente sorteado):
es lo que un `_04`/`_05` puede contrastar contra el gating estimado del
PSBPM-FD (mupsij/taupsij, apij/bpij) para verificar si la mezcla probit
recupera la dependencia real.
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
from .sim_escenario_B1 import ConfigEscenarioB1, funciones_rasgo, medias_mezcla, simular_mezcla_curvas
from .sim_escenario_B2 import B_MECANISMOS_B2, transformar_rasgos_B2
from .sim_series_clasicas import r2_empirico_media_condicional

__all__ = [
    "A_ASIGNACION_B3",
    "probabilidades_softmax",
    "ConfigEscenarioB3",
    "generar_escenario_B3",
    "resumen_escenario_B3",
]


# ==========================================================================
# ASIGNACION DEPENDIENTE (SOFTMAX)
# ==========================================================================

A_ASIGNACION_B3 = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.8, 0.0],
    [-1.0, 0.0, 0.8],
])
"""
Matriz (mecanismo x rasgo) de q_k(R) = A_ASIGNACION_B3[k] . R. Fila 0 (q_1) es
la referencia del softmax; ver docstring del modulo.
"""


def probabilidades_softmax(R: np.ndarray, A: np.ndarray = A_ASIGNACION_B3) -> np.ndarray:
    """p_k(R) = softmax_k(A[k] . R), estabilizado restando el maximo."""
    q = A @ R
    q = q - q.max()
    p = np.exp(q)
    return p / p.sum()


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioB3(ConfigEscenarioB1):
    """
    Algoritmo B-3: mismos parametros que B-1/B-2 (pi, sigma_eps, ell,
    prop_train_referencia). `pi` queda sin uso en la generacion (la asignacion
    depende de R_{t-1} via `probabilidades_softmax`, no de `cfg.pi`); se
    conserva por compatibilidad de la dataclass y porque el diagnostico lo usa
    como referencia de comparacion (frecuencia empirica de cada mecanismo
    marginalizada sobre R, que en general NO coincide con pi cuando la
    asignacion depende de la curva anterior).
    """


# ==========================================================================
# GENERADOR B-3
# ==========================================================================

def generar_escenario_B3(cfg: ConfigEscenarioB3,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """
    Genera R replicas del Algoritmo B-3: mismos mecanismos no lineales de
    B-2, con la probabilidad de cada mecanismo dependiendo de los rasgos de la
    curva anterior via `probabilidades_softmax`.
    """
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    pesos = pesos_trapezoidales(tau)
    Phi = funciones_rasgo(tau)
    MU = medias_mezcla(tau)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)

    def pi_efectiva(R_prev: np.ndarray) -> np.ndarray:
        return probabilidades_softmax(R_prev)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)
    R_, T, L = int(cfg.R), int(cfg.T), int(cfg.L)
    observaciones = np.empty((R_, T, L))
    curvas = np.empty((R_, T, L))
    mc = np.empty((R_, T, L))
    Z_all = np.empty((R_, T), dtype=int)
    Rasgos_all = np.empty((R_, T, 3))
    Prob_all = np.empty((R_, T, 3))

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
        Prob_all[r] = np.stack([probabilidades_softmax(R_r[t]) for t in range(T)])

    pi_marginal = np.asarray(cfg.pi, dtype=float)   # solo referencia (ver docstring de la config)
    media = pi_marginal @ MU
    internos = {
        "funciones_rasgo": Phi,
        "pesos_cuadratura": pesos,
        "medias_mecanismos": MU,
        "matriz_coeficientes": B_MECANISMOS_B2,
        "matriz_asignacion": A_ASIGNACION_B3,
        "cov_innovacion": K,
        "pi_referencia": pi_marginal,
        "mecanismo": Z_all,
        "rasgos": Rasgos_all,
        "probabilidades_mecanismo": Prob_all,
        "media_condicional": mc,
    }
    salida = SalidaSimulacion(
        observaciones=observaciones, curvas=curvas, grilla=tau, media=media,
        semillas=registro, config=cfg, internos=internos,
    )
    if diagnosticar:
        salida.diagnostico = resumen_escenario_B3(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def resumen_escenario_B3(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del B-3: diagnostico comun, frecuencia empirica de cada
    mecanismo (que aqui NO tiene por que igualar pi_referencia: la asignacion
    depende de R), acotamiento de los rasgos, R^2 del oraculo dependiente, y la
    dispersion de `probabilidades_mecanismo` (que tan lejos de (1/3,1/3,1/3)
    mueve el softmax en la practica: si quedara siempre cerca del centro, la
    dependencia seria cosmetica).
    """
    if not isinstance(salida.config, ConfigEscenarioB3):
        raise TypeError("resumen_escenario_B3 requiere ConfigEscenarioB3; se "
                        f"recibio {type(salida.config).__name__}.")
    Z = salida.internos["mecanismo"]
    Rasgos = salida.internos["rasgos"]
    P = salida.internos["probabilidades_mecanismo"]
    frac = np.array([float((Z == k).mean()) for k in range(3)])

    return {
        **diagnostico_comun(salida),
        "pi_referencia": salida.internos["pi_referencia"].tolist(),
        "pi_empirica": frac.tolist(),
        "prob_mecanismo_media": P.reshape(-1, 3).mean(axis=0).tolist(),
        "prob_mecanismo_desviacion_del_centro": float(
            np.abs(P.reshape(-1, 3) - 1.0 / 3.0).mean()),
        "max_abs_rasgos": float(np.max(np.abs(Rasgos))),
        "var_rasgos": Rasgos.reshape(-1, 3).var(axis=0).tolist(),
        "r2_oraculo_1rezago": r2_empirico_media_condicional(salida),
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "T0_referencia": int(np.floor(salida.config.prop_train_referencia * salida.config.T)),
    }
