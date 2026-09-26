"""
sim_escenario_C2.py
====================
Algoritmo C-2 del anexo (`docs/01 Anexo.tex`, `ane_00_03_02_alg_c2`): la
misma asignacion de C-1 --softmax de q_k lineales (afines) sobre los scores de
los L = 2 rezagos, con los MISMOS coeficientes `C_ASIG_C1`, `G_ASIG_C1`-- pero
con dinamicas cuya ESTRUCTURA difiere entre mecanismos:

    xi_t = f_{Z_t}(xi_{t-1}, xi_{t-2}) + eps_t.

Lo que se interpreta del anexo
------------------------------
- f_k LINEALES (afines). El anexo no dice que f_k sea no lineal en C-2, y
  reserva la no linealidad de la dinamica para C-3 ("este algoritmo suma la no
  linealidad"). Lo que cambia respecto de C-1 es el PATRON de no nulos de
  A_{k,l}: que rezagos y que scores intervienen, no solo sus valores.
- f_k incluye la localizacion del mecanismo: se conservan las mu~_k de C-1
  (`M_TILDE_C1`), de modo que C-2 difiere de C-1 SOLO en la dinamica, como B-2
  difiere de B-1 solo en los mecanismos.
- mu(tau), phi_j, espectro nominal, bloque pasivo y calentamiento: los de C-1
  (ver docstring de `sim_escenario_C1`). La sd de la innovacion se recalibra
  (`SIGMA_TILDE_C2`) para que Var(z_j) ~ 1 en j = 2..4, como en C-1.

Diseno (coordenadas z, (respuesta <- covariable))
-------------------------------------------------
    k = 1 (centro, "inercial"): solo rezago 1, diagonal 0.3 en 1..4 mas
          (3 <- 2) = 0.6.
    k = 2 (nivel alto, "rezago 2"): solo rezago 2: (1 <- 1) = 0.3,
          (2 <- 1) = 0.6, (3 <- 3) = 0.4, (4 <- 2) = -0.5.
    k = 3 (nivel bajo, "cruzado en rezago 1"): (1 <- 1) = 0.3, (2 <- 1) = -0.6,
          (2 <- 3) = 0.4, (3 <- 4) = 0.5, (4 <- 4) = 0.3.

El AR propio de xi_1 en los extremos queda en 0.3 por la misma razon que en
C-1 (la permanencia la pone la asignacion; con 0.7 los regimenes se vuelven
absorbentes con el salto delta = 1 de `M_TILDE_C1`).

Es el ejemplo del anexo hecho explicito: un mecanismo depende del primer
rezago, otro del segundo y otro de una combinacion cruzada distinta. (2 <- 1)
vuelve a cambiar de signo entre los extremos, pero ademas cambia de REZAGO
(2 en el alto, 1 en el bajo): la dependencia de xi_2 en xi_1 no la describe
ninguna matriz comun.

Cifras de calibracion
---------------------
Tres fuentes: (a) "poblacional", 10 replicas de T = 3000 sin ruido (20 x
10000 para estabilidad); (b) la realizacion de las corridas vivas, T = 1000,
seed 41232; (c) el pipeline del _01 sobre (b) --GCV, FPCA en train, sigma_obs
= 0.25--, medido con un script suelto. R^2 en L^2 = agregado sobre las J
componentes; "lineal" = MCO sobre los L rezagos de las J componentes.

    ocupacion (centro, alto, bajo)   (a) 0.46 0.26 0.28    (b) 0.47 0.22 0.32
    racha media                      (a) 3.3  3.7  4.0     (b) 3.4  3.4  4.2
    transiciones en T = 1000         (b) 277, 75 de ellas en el bloque de prueba
    R^2 L^2 oraculo / lineal         (a) 0.417 / 0.381     (b) 0.394 / 0.369
    R^2 por componente (a), xi_1..4  oraculo 0.67 0.21 0.20 0.11
                                     lineal  0.67 0.12 0.12 0.09
    R^2 sobre scores ESTIMADOS (c)   oraculo 0.64 0.22 0.14 0.10
                                     lineal  0.63 0.15 0.07 0.09
    varianza entre mecanismos / total, xi_1..4 (a)   0.16 0.07 0.09 0.03
    corr. contemporanea max. (bloque activo) (a)     0.11
    |cos| ejes vs autovectores de Cov(xi) (a)        >= 0.984
    |cos(psi_k, phi_k)|, FPCA estimada (c), k=1..6   0.97 0.95 0.96 0.99 0.99 0.99
    GCV (c): n_basis = 14, orden 4 (K = 14); regla del 95 %: M = 5
    radio espectral de la companera por mecanismo    0.30 0.63 0.30
    sup |xi| bloque activo en 200 000 pasos          2.6
    varianza por bloques de 1000, min / max sobre la media   0.85 / 1.13

En la realizacion (c) las tres primeras autofunciones estimadas rotan algo
respecto de phi_1..phi_3 (|cos| 0.95-0.97, ~15-19 grados): la correlacion
contemporanea de esa muestra (0.12) es mayor que la poblacional. El contraste
de PIP del _03 contra `soporte` es aproximado en esas componentes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sim_comun import SalidaSimulacion
from .sim_escenario_C1 import (
    C_ASIG_C1,
    G_ASIG_C1,
    K_MECANISMOS,
    M_TILDE_C1,
    ConfigEscenarioC1,
    MecanismoScores,
    _matrices,
    generar_mezcla_scores,
    mecanismo_lineal,
    resumen_mezcla_scores,
    sigma_tilde,
)

__all__ = [
    "A_TILDE_C2",
    "SIGMA_TILDE_C2",
    "ConfigEscenarioC2",
    "mecanismo_C2",
    "generar_escenario_C2",
    "resumen_escenario_C2",
]


A_TILDE_C2 = _matrices({
    0: [(1, 1, 1, 0.3), (1, 2, 2, 0.3), (1, 3, 3, 0.3), (1, 4, 4, 0.3), (1, 3, 2, 0.6)],
    1: [(2, 1, 1, 0.3), (2, 2, 1, 0.6), (2, 3, 3, 0.4), (2, 4, 2, -0.5)],
    2: [(1, 1, 1, 0.3), (1, 2, 1, -0.6), (1, 2, 3, 0.4), (1, 3, 4, 0.5), (1, 4, 4, 0.3)],
}, 2)
"""A~_{k,l} (K, L, J, J) de C-2: el patron cambia con el mecanismo."""
assert A_TILDE_C2.shape[0] == K_MECANISMOS

SIGMA_TILDE_C2 = sigma_tilde((0.45, 0.87, 0.9, 0.94))
"""sd de eps en z; Var(z_j) ~ 1 en j = 2..4, como en C-1."""


@dataclass
class ConfigEscenarioC2(ConfigEscenarioC1):
    """Algoritmo C-2: mismos parametros que C-1 (L = 2 rezagos)."""


def mecanismo_C2() -> MecanismoScores:
    return mecanismo_lineal(M_TILDE_C1, A_TILDE_C2, C_ASIG_C1, G_ASIG_C1, SIGMA_TILDE_C2)


def generar_escenario_C2(cfg: ConfigEscenarioC2) -> SalidaSimulacion:
    """R replicas del Algoritmo C-2."""
    return generar_mezcla_scores(cfg, mecanismo_C2(), resumen_escenario_C2)


def resumen_escenario_C2(salida: SalidaSimulacion) -> dict:
    if not isinstance(salida.config, ConfigEscenarioC2):
        raise TypeError("resumen_escenario_C2 requiere ConfigEscenarioC2; se "
                        f"recibio {type(salida.config).__name__}.")
    return resumen_mezcla_scores(salida)
