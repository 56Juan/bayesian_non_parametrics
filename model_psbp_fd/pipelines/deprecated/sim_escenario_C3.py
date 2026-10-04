"""
sim_escenario_C3.py
====================
Algoritmo C-3 del anexo (`docs/01 Anexo.tex`, `ane_00_03_03_alg_c3`): mezcla de
K = 3 mecanismos con dinamica NO LINEAL y asignacion NO LINEAL, ambas con
interacciones entre scores de distintos rezagos, sobre L = 3 rezagos:

    P(Z_t = k | xi_{t-1..t-3}) = softmax_k q_k(xi_{t-1}, xi_{t-2}, xi_{t-3}),
    xi_t = f_{Z_t}(xi_{t-1}, xi_{t-2}, xi_{t-3}) + eps_t.

mu(tau), phi_j, espectro nominal, bloque pasivo y calentamiento son los de C-1
(ver docstring de `sim_escenario_C1`); cambian f_k, q_k y la sd de la
innovacion, recalibrada (`SIGMA_TILDE_C3`) para que Var(z_j) ~ 1 en j = 2..4.

Lo que se interpreta del anexo
------------------------------
El anexo pide "terminos no lineales e interacciones entre los scores de
distintos rezagos" en f_k y q_k, sin forma. Se eligen:

- En f_k, no linealidades ACOTADAS: tanh(u), tanh(u v) y tanh(u) tanh(v), con u
  y v scores de rezagos distintos, sobre una parte lineal estable. Un termino
  cuadratico sin acotar retroalimenta mas fuerte cuanto mas grande es el score
  y, en una mezcla con asignacion dependiente del estado, no hay condicion
  cerrada de estacionariedad; con perturbaciones acotadas sobre una parte
  lineal contractiva el proceso queda acotado en distribucion. Igual se MIDE
  (sup |xi| y varianza por bloques en `resumen_mezcla_scores`).
- En q_k, una parte direccional lineal (que decide alto contra bajo), una
  interaccion z_{2,t-1} z_{3,t-2} y un termino cuadratico en el cambio
  z_{1,t-1} - z_{1,t-3} que devuelve al centro cuando el nivel se movio mucho
  en tres periodos. En el softmax no hace falta acotar nada.

Diseno (coordenadas z; `PARAMETROS_C3`)
-----------------------------------------
    q_1 = 0
    q_2 = c + g s + eta z_{2,t-1} z_{3,t-2} - kappa (z_{1,t-1} - z_{1,t-3})^2
    q_3 = c - g s + eta z_{2,t-1} z_{3,t-2} - kappa (z_{1,t-1} - z_{1,t-3})^2
    s = z_{1,t-1} + 0.4 z_{1,t-2},  c = -2.1, g = 2.2, eta = 1.0, kappa = 0.15

    k = 1 (centro, interacciones):
        z1 = 0.2 z_{1,t-1} + b tanh(z_{2,t-1} z_{3,t-2})
        z2 = 0.3 z_{2,t-1} + 0.6 tanh(z_{1,t-3})
        z3 = 0.3 z_{3,t-1} - w tanh(z_{2,t-2}) tanh(z_{4,t-1})
        z4 = 0.3 z_{4,t-1}
    k = 2 (nivel alto, saturacion):
        z1 = delta + 0.15 z_{1,t-1} + 0.15 tanh(z_{1,t-2})
        z2 = 0.3 z_{2,t-1} + b tanh(z_{1,t-1} z_{1,t-2}) - 0.4
        z3 = 0.3 z_{3,t-1} + w tanh(z_{2,t-3})
        z4 = 0.2 z_{4,t-1} - w tanh(z_{3,t-1} z_{1,t-3})
    k = 3 (nivel bajo, umbral):
        z1 = -delta + 0.15 z_{1,t-1} + 0.15 tanh(z_{1,t-2})
        z2 = 0.3 z_{2,t-1} + b tanh(z_{1,t-1} z_{1,t-2}) - 0.4
        z3 = 0.3 z_{3,t-1} + w tanh(z_{4,t-3})
        z4 = 0.2 z_{4,t-1} + w (|tanh(z_{2,t-2})| - 0.5)

    delta = 1.0, b = 0.8, w = 0.9.

Como en C-1, el salto delta es grande y la persistencia propia de xi_1 en los
extremos chica (0.15 + 0.15 tanh): la permanencia la pone la asignacion y la
ley de xi_1 cerca de un cambio es bimodal. w = 0.9 (y no 0.6) para que los
mecanismos se distingan tambien en xi_3 y xi_4: con 0.6 su fraccion de
varianza entre mecanismos era 0.03 y 0.01.

tanh(z_{1,t-1} z_{1,t-2}) es positivo cuando el nivel persiste en cualquiera de
los dos extremos: xi_2 responde a la persistencia del nivel, no a su signo, y
por eso el termino lleva el MISMO signo en los mecanismos 2 y 3 (con signos
opuestos xi_2 quedaba correlacionada con xi_1 --corr 0.39-- y la FPCA rotaba
las dos primeras autofunciones). El -0.4 centra ese termino.

Cifras de calibracion
---------------------
Tres fuentes: (a) "poblacional", 10 replicas de T = 3000 sin ruido (20 x
10000 para estabilidad); (b) la realizacion de las corridas vivas, T = 1000,
seed 41232; (c) el pipeline del _01 sobre (b) --GCV, FPCA en train, sigma_obs
= 0.25--, medido con un script suelto. R^2 en L^2 = agregado sobre las J
componentes; "lineal" = MCO sobre los L rezagos de las J componentes.

    ocupacion (centro, alto, bajo)   (a) 0.38 0.21 0.41    (b) 0.41 0.19 0.41
    racha media                      (a) 2.7  4.4  4.6     (b) 2.9  3.7  4.3
    transiciones en T = 1000         (b) 286, 78 de ellas en el bloque de prueba
    R^2 L^2 oraculo / lineal         (a) 0.438 / 0.389     (b) 0.412 / 0.376
    R^2 por componente (a), xi_1..4  oraculo 0.68 0.20 0.29 0.16
                                     lineal  0.63 0.16 0.19 0.09
    R^2 sobre scores ESTIMADOS (c)   oraculo 0.65 0.20 0.23 0.15
                                     lineal  0.60 0.18 0.15 0.10
    varianza entre mecanismos / total, xi_1..4 (a)   0.15 0.05 0.07 0.02
    corr. contemporanea max. (bloque activo) (a)     0.07
    |cos| ejes vs autovectores de Cov(xi) (a)        >= 0.994
    |cos(psi_k, phi_k)|, FPCA estimada (c), k=1..6   0.98 0.98 1.00 1.00 0.99 0.99
    GCV (c): n_basis = 14, orden 4 (K = 14); regla del 95 %: M = 6
             (var. acumulada en M = 5: 0.947)
    radio espectral de la linealizacion en 0         0.30 0.47 0.47
    sup |xi| bloque activo en 200 000 pasos          2.5
    varianza por bloques de 1000, min / max sobre la media   0.85 / 1.12

La asimetria de ocupacion entre alto (0.21) y bajo (0.41) es del diseno: el
termino -0.4 de xi_2 y la interaccion eta z_2 z_3 de la asignacion no son
simetricos en el signo de xi_1. Ningun mecanismo baja del 10 %. A diferencia
de C-1 y C-2, aqui xi_1 SI tiene brecha no lineal (0.68 contra 0.63): la
saturacion tanh y el termino kappa de la asignacion no los aproxima ninguna
recta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim_comun import SalidaSimulacion
from ..sim_escenario_C1 import (
    AR_PASIVO,
    BLOQUE_ACTIVO,
    J_SCORES,
    K_MECANISMOS,
    sigma_tilde,
    ConfigEscenarioC1,
    MecanismoScores,
    generar_mezcla_scores,
    resumen_mezcla_scores,
)

__all__ = [
    "PARAMETROS_C3",
    "SIGMA_TILDE_C3",
    "ConfigEscenarioC3",
    "mecanismo_C3",
    "generar_escenario_C3",
    "resumen_escenario_C3",
]

PARAMETROS_C3 = {"delta": 1.0, "b": 0.8, "w": 0.9,
                 "c": -2.1, "g": 2.2, "eta": 1.0, "kappa": 0.15}
"""Constantes de f~_k y q~_k de C-3 (ver docstring del modulo)."""

SIGMA_TILDE_C3 = sigma_tilde((0.45, 0.88, 0.8, 0.89))
"""sd de eps en z; Var(z_j) ~ 1 en j = 2..4, como en C-1."""

_L3 = 3
th = np.tanh


def _medias_C3(x: np.ndarray, p: dict = PARAMETROS_C3) -> np.ndarray:
    """f~_k(x) (K, J); x (3, J) con x[l-1] = z_{t-l}."""
    z1, z2, z3, z4 = (x[:, j] for j in range(BLOQUE_ACTIVO))
    d, b, w = p["delta"], p["b"], p["w"]
    F = np.zeros((K_MECANISMOS, x.shape[1]))
    F[:, BLOQUE_ACTIVO:] = AR_PASIVO * x[0, BLOQUE_ACTIVO:]
    F[0, 0] = 0.2 * z1[0] + b * th(z2[0] * z3[1])
    F[0, 1] = 0.3 * z2[0] + 0.6 * th(z1[2])
    F[0, 2] = 0.3 * z3[0] - w * th(z2[1]) * th(z4[0])
    F[0, 3] = 0.3 * z4[0]
    F[1, 0] = d + 0.15 * z1[0] + 0.15 * th(z1[1])
    F[1, 1] = 0.3 * z2[0] + b * th(z1[0] * z1[1]) - 0.4
    F[1, 2] = 0.3 * z3[0] + w * th(z2[2])
    F[1, 3] = 0.2 * z4[0] - w * th(z3[0] * z1[2])
    F[2, 0] = -d + 0.15 * z1[0] + 0.15 * th(z1[1])
    F[2, 1] = 0.3 * z2[0] + b * th(z1[0] * z1[1]) - 0.4
    F[2, 2] = 0.3 * z3[0] + w * th(z4[2])
    F[2, 3] = 0.2 * z4[0] + w * (np.abs(th(z2[1])) - 0.5)
    return F


def _asignacion_C3(x: np.ndarray, p: dict = PARAMETROS_C3) -> np.ndarray:
    """q~_k(x) (K,)."""
    z1, z2, z3 = x[:, 0], x[:, 1], x[:, 2]
    s = z1[0] + 0.4 * z1[1]
    comun = p["c"] + p["eta"] * z2[0] * z3[1] - p["kappa"] * (z1[0] - z1[2]) ** 2
    return np.array([0.0, comun + p["g"] * s, comun - p["g"] * s])


def _soporte_C3() -> tuple[np.ndarray, np.ndarray]:
    """(soporte_dinamica (K, J, L, J), soporte_asignacion (L, J)), leidos de f~ y q~."""
    Sd = np.zeros((K_MECANISMOS, J_SCORES, _L3, J_SCORES), dtype=bool)
    # (k, respuesta, rezago, covariable), base 1
    pares = {
        0: [(1, 1, 1), (1, 1, 2), (1, 2, 3), (2, 1, 2), (2, 3, 1),
            (3, 1, 3), (3, 2, 2), (3, 1, 4), (4, 1, 4)],
        1: [(1, 1, 1), (1, 2, 1), (2, 1, 2), (2, 1, 1), (2, 2, 1),
            (3, 1, 3), (3, 3, 2), (4, 1, 4), (4, 1, 3), (4, 3, 1)],
        2: [(1, 1, 1), (1, 2, 1), (2, 1, 2), (2, 1, 1), (2, 2, 1),
            (3, 1, 3), (3, 3, 4), (4, 1, 4), (4, 2, 2)],
    }
    for k, lista in pares.items():
        for j, l, m in lista:
            Sd[k, j - 1, l - 1, m - 1] = True
        for j in range(BLOQUE_ACTIVO, J_SCORES):
            Sd[k, j, 0, j] = True
    Sq = np.zeros((_L3, J_SCORES), dtype=bool)
    for l, m in [(1, 1), (2, 1), (3, 1), (1, 2), (2, 3)]:
        Sq[l - 1, m - 1] = True
    return Sd, Sq


def _jacobiano_en_cero(h: float = 1e-6) -> np.ndarray:
    """d f~_k / d x en x = 0, (K, L, J, J), por diferencias centradas."""
    Jac = np.zeros((K_MECANISMOS, _L3, J_SCORES, J_SCORES))
    for l in range(_L3):
        for m in range(J_SCORES):
            e = np.zeros((_L3, J_SCORES))
            e[l, m] = h
            Jac[:, l, :, m] = (_medias_C3(e) - _medias_C3(-e)) / (2 * h)
    return Jac


@dataclass
class ConfigEscenarioC3(ConfigEscenarioC1):
    """Algoritmo C-3: mismos parametros que C-1, con L = 3 rezagos."""

    n_rezagos: int = 3
    N_REZAGOS_ANEXO = 3


def mecanismo_C3() -> MecanismoScores:
    Sd, Sq = _soporte_C3()
    difiere = np.zeros(J_SCORES, dtype=bool)
    difiere[:BLOQUE_ACTIVO] = True
    return MecanismoScores(
        medias=_medias_C3, asignacion=_asignacion_C3, sigma=SIGMA_TILDE_C3.copy(),
        n_rezagos=_L3, soporte_dinamica=Sd, soporte_asignacion=Sq,
        difiere=difiere, jacobiano=_jacobiano_en_cero())


def generar_escenario_C3(cfg: ConfigEscenarioC3) -> SalidaSimulacion:
    """R replicas del Algoritmo C-3."""
    return generar_mezcla_scores(cfg, mecanismo_C3(), resumen_escenario_C3)


def resumen_escenario_C3(salida: SalidaSimulacion) -> dict:
    if not isinstance(salida.config, ConfigEscenarioC3):
        raise TypeError("resumen_escenario_C3 requiere ConfigEscenarioC3; se "
                        f"recibio {type(salida.config).__name__}.")
    return resumen_mezcla_scores(salida)
