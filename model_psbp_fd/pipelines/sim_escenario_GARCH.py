"""
sim_escenario_GARCH.py
======================
Escenario de heterocedasticidad condicional en los scores (corrida 201, algoritmo C-1 del
anexo). Cada score activo j = 1..4 sigue un AR(1) con innovaciones GARCH(1,1):

    u_tj = phi u_{t-1,j} + a_tj,   a_tj = sqrt(h_tj) e_tj,   e_tj ~ N(0, 1),
    h_tj = alpha_0 + alpha_1 a_{t-1,j}^2 + beta_1 h_{t-1,j},   alpha_0 = sigma_a^2 (1 - alpha_1 - beta_1),

con los valores del anexo (phi = 0.5, alpha_1 = 0.25, beta_1 = 0.60, sigma_a^2 = 1). La media
condicional es LINEAL (phi u_{t-1}) y la no linealidad esta solo en la varianza, cuya memoria
es de 1/(1 - alpha_1 - beta_1) = 6.7 periodos, que aqui son curvas. La combinacion alpha_1
alto / beta_1 moderado da una senal de volatilidad fuerte y aprendible desde los rezagos (acf1 de
los cuadrados ~ 0.36, mediana 0.28 con T = 1000, curtosis teorica 5.5).

Que ve un modelo con covariables de rezago propio: h_t depende de la historia infinita de
innovaciones a traves de beta_1, que no es observable; pero la innovacion reciente es
a_{t-1} = u_{t-1} - phi u_{t-2}, funcion de los DOS rezagos propios mas proximos, de modo que
con n_rezagos >= 2 el termino alpha_1 a_{t-1}^2 si puede aprenderse. Por eso n_rezagos = 2.

Score: z_tj = u_tj sqrt(1 - phi^2) / sigma_a (varianza marginal unitaria), xi_tj = sqrt(lambda_j)
z_tj. Los scores 5..10 son AR(1) pasivos (esquema comun, `sim_scores_comun`).
"""

from __future__ import annotations

import numpy as np

from .sim_scores_comun import (
    AR_PASIVO, BLOQUE_ACTIVO, J_SCORES, generar_desde_scores, resumen_scores,
)
from .sim_escenario_TAR import ConfigEscenarioTAR

__all__ = ["PARAMETROS_GARCH", "ConfigEscenarioGARCH", "generar_escenario_GARCH",
           "resumen_escenario_GARCH"]

PARAMETROS_GARCH = {"phi": 0.5, "alpha1": 0.25, "beta1": 0.60, "sigma_a2": 1.0}


class ConfigEscenarioGARCH(ConfigEscenarioTAR):
    """Mismos defaults de observacion que el TAR (sigma_obs = 0, n_rezagos = 2)."""


def _simular(T: int, burn_in: int, rng: np.random.Generator) -> dict:
    J, A = J_SCORES, BLOQUE_ACTIVO
    phi, a1, b1, s2 = (PARAMETROS_GARCH[k] for k in ("phi", "alpha1", "beta1", "sigma_a2"))
    assert a1 + b1 < 1.0, "GARCH no estacionario: alpha_1 + beta_1 >= 1."
    a0 = s2 * (1.0 - a1 - b1)
    k_z = np.sqrt(1.0 - phi ** 2) / np.sqrt(s2)           # u -> z (varianza marginal unitaria)
    sig_p = np.sqrt(1.0 - AR_PASIVO ** 2)
    n = int(burn_in) + int(T)
    z = np.zeros((n, J)); ora = np.zeros((n, J)); hz = np.zeros((n, A))
    u, a_prev, h_prev = np.zeros(A), np.zeros(A), np.full(A, s2)
    zp = np.zeros(J - A)
    for i in range(n):
        e = rng.standard_normal(J)
        h = a0 + a1 * a_prev ** 2 + b1 * h_prev
        a = np.sqrt(h) * e[:A]
        ora[i, :A] = phi * u * k_z                          # media condicional en z
        u = phi * u + a
        z[i, :A] = u * k_z
        hz[i] = h * k_z ** 2                                 # varianza condicional en z
        ora[i, A:] = AR_PASIVO * zp
        zp = AR_PASIVO * zp + sig_p * e[A:]
        z[i, A:] = zp
        a_prev, h_prev = a, h
    c = int(burn_in)
    return {"z": z[c:], "oraculo": ora[c:], "var_condicional": hz[c:]}


def generar_escenario_GARCH(cfg: ConfigEscenarioGARCH):
    """R replicas del escenario GARCH en los scores."""
    salida = generar_desde_scores(cfg, _simular, "GARCH", parametros=PARAMETROS_GARCH)
    salida.diagnostico = resumen_escenario_GARCH(salida)
    return salida


def resumen_escenario_GARCH(salida) -> dict:
    """`resumen_scores` mas la memoria nominal de la volatilidad y la curtosis de los scores activos."""
    d = resumen_scores(salida)
    p = PARAMETROS_GARCH
    d["memoria_volatilidad"] = 1.0 / (1.0 - p["alpha1"] - p["beta1"])
    xi = salida.internos["scores"][..., :BLOQUE_ACTIVO]
    x = xi.reshape(-1, BLOQUE_ACTIVO)
    d["curtosis_scores_activos"] = (((x - x.mean(0)) ** 4).mean(0) / x.var(0) ** 2).tolist()
    return d
