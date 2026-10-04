"""
sim_escenario_TAR3.py
=====================
Escenario TAR de TRES regimenes sobre los scores (corrida 202, algoritmo C-2 del
anexo). Cada score activo j = 1..4 evoluciona en una coordenada cruda u_j con tres
regimenes ordenados, asignados por un probit ordenado sobre su PROPIO rezago:

    P_r = Phi((u_{j,t-1} - c_rj) / h_j),                     r = 1, 2,
    P(R=1) = 1 - P_1,  P(R=2) = P_1 - P_2,  P(R=3) = P_2,
    u_tj = mu_kj + phi_kj u_{j,t-1} + s_kj e_tj              si R_tj = k.

Con h -> 0 es un SETAR duro; h > 0 hace la transicion suave y reproduce la forma del
peso probit del stick-breaking. Como el regimen depende solo del rezago propio, la
informacion que lo determina es la que usa un modelo con covariables de rezago propio.

Calibracion (`PARAMETROS_TAR3`)
-------------------------------
Una busqueda por simulacion larga (un score a la vez) con umbrales en los terciles de la
distribucion estacionaria; se conservo, entre las soluciones con ocupacion minima de 15 %
por regimen, la de mayor brecha entre el oraculo de Bayes y el mejor AR lineal con racha
media del regimen de al menos 2.5 periodos. Las cifras realizadas estan en el diagnostico
(`resumen_scores`): ocupacion y racha de cada regimen, R^2 del oraculo contra el lineal.

Score: z_tj = (u_tj - m_j) / d_j con (m_j, d_j) la media y sd POBLACIONALES de u_j
(`MOMENTOS_TAR3`, de `momentos_poblacionales`), y xi_tj = sqrt(lambda_j) z_tj. Los scores
5..10 son AR(1) pasivos (esquema comun, `sim_scores_comun`).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .sim_scores_comun import (
    AR_PASIVO, BLOQUE_ACTIVO, J_SCORES, generar_desde_scores, resumen_scores,
)
from .sim_escenario_TAR import ConfigEscenarioTAR

__all__ = ["PARAMETROS_TAR3", "MOMENTOS_TAR3", "ConfigEscenarioTAR3", "momentos_poblacionales",
           "generar_escenario_TAR3", "resumen_escenario_TAR3"]

K_REGIMENES = 3

# Por score activo: umbrales (A, 2), ancho (A,), nivel / coeficiente / escala por regimen (A, 3).
C_TAR3 = np.array([
    [-0.7144, -0.2100],
    [+0.4928, +1.9766],
    [+0.3353, +2.6542],
    [-0.5248, -0.0641],
])
H_TAR3 = np.array([0.0565, 0.1378, 0.2839, 0.0643])
MU_TAR3 = np.array([
    [+0.2924, -0.6889, +0.5583],
    [-0.0867, +3.4271, +0.1749],
    [-0.7391, +3.1718, -0.1185],
    [-1.4294, -0.1196, +1.2451],
])
PHI_TAR3 = np.array([
    [+0.5672, +0.6932, -0.4270],
    [+0.5671, -0.6985, +0.1514],
    [-0.8432, -0.4001, +0.2210],
    [-0.6124, +0.1795, -0.8070],
])
S_TAR3 = np.array([
    [+0.3306, +0.4774, +0.4162],
    [+0.4539, +0.4291, +0.3515],
    [+0.3412, +0.2508, +0.4694],
    [+0.4937, +0.3604, +0.5570],
])
PARAMETROS_TAR3 = {"c": C_TAR3, "h": H_TAR3, "mu": MU_TAR3, "phi": PHI_TAR3, "s": S_TAR3}
# (media, sd) poblacionales de u_j: momentos_poblacionales(T=200_000, seed=7).
MOMENTOS_TAR3 = np.array([
    [+0.0199, 0.6976],
    [+0.3999, 1.2400],
    [+0.8894, 1.4249],
    [+0.1441, 0.8767],
])


class ConfigEscenarioTAR3(ConfigEscenarioTAR):
    """Mismos defaults de observacion que el TAR de dos regimenes (sigma_obs = 0)."""


def _paso(u_prev, unif, e):
    """Un paso de los scores activos en coordenada u. Devuelve u, regimen (0..2),
    probabilidades (A, 3) y media condicional exacta (A,)."""
    A = BLOQUE_ACTIVO
    P = norm.cdf((u_prev[:, None] - C_TAR3) / H_TAR3[:, None])
    p = np.column_stack([1 - P[:, 0], P[:, 0] - P[:, 1], P[:, 1]])
    reg = np.minimum((unif[:, None] > np.cumsum(p, axis=1)).sum(axis=1), K_REGIMENES - 1)
    m = MU_TAR3 + PHI_TAR3 * u_prev[:, None]
    ar = np.arange(A)
    return m[ar, reg] + S_TAR3[ar, reg] * e, reg, p, (p * m).sum(axis=1)


def momentos_poblacionales(T: int = 200_000, seed: int = 7) -> np.ndarray:
    """(media, sd) de u_j en una simulacion larga: la fuente de MOMENTOS_TAR3."""
    rng = np.random.default_rng(seed)
    u = np.zeros(BLOQUE_ACTIVO)
    U = np.empty((T, BLOQUE_ACTIVO))
    for t in range(T):
        u = _paso(u, rng.random(BLOQUE_ACTIVO), rng.standard_normal(BLOQUE_ACTIVO))[0]
        U[t] = u
    U = U[1000:]
    return np.column_stack([U.mean(0), U.std(0)])


def _simular(T: int, burn_in: int, rng: np.random.Generator) -> dict:
    J, A = J_SCORES, BLOQUE_ACTIVO
    m, d = MOMENTOS_TAR3[:, 0], MOMENTOS_TAR3[:, 1]
    sig_p = np.sqrt(1.0 - AR_PASIVO ** 2)
    n = int(burn_in) + int(T)
    z = np.zeros((n, J)); ora = np.zeros((n, J))
    reg = np.zeros((n, J), dtype=int); P = np.zeros((n, J, K_REGIMENES))
    u, zp = m.copy(), np.zeros(J - A)
    for i in range(n):
        unif, e = rng.random(A), rng.standard_normal(J)
        u_prev = u
        u, r, p, o = _paso(u_prev, unif, e[:A])
        z[i, :A], reg[i, :A], P[i, :A] = (u - m) / d, r, p
        ora[i, :A] = (o - m) / d
        ora[i, A:] = AR_PASIVO * zp
        zp = AR_PASIVO * zp + sig_p * e[A:]
        z[i, A:] = zp
    c = int(burn_in)
    return {"z": z[c:], "oraculo": ora[c:], "regimen": reg[c:], "p_regimen": P[c:]}


def generar_escenario_TAR3(cfg: ConfigEscenarioTAR3):
    """R replicas del escenario TAR de tres regimenes."""
    assert np.all(MOMENTOS_TAR3[:, 1] > 0), "MOMENTOS_TAR3 sin calcular: correr momentos_poblacionales()."
    salida = generar_desde_scores(cfg, _simular, "TAR3", parametros=PARAMETROS_TAR3)
    salida.diagnostico = resumen_escenario_TAR3(salida)
    return salida


def resumen_escenario_TAR3(salida) -> dict:
    """`resumen_scores` mas la ocupacion y la racha media de cada regimen por score activo."""
    d = resumen_scores(salida)
    REG = salida.internos["regimen"][..., :BLOQUE_ACTIVO]          # (R, T, A)
    K = K_REGIMENES
    occ, rach = [], []
    for j in range(BLOQUE_ACTIVO):
        o, r = np.zeros(K), np.zeros(K)
        for rep in range(REG.shape[0]):
            x = REG[rep, :, j]
            o += np.bincount(x, minlength=K) / len(x)
            cambios = np.flatnonzero(np.diff(x) != 0)
            fin = np.r_[cambios, len(x) - 1]
            largos = np.diff(np.r_[-1, fin])
            r += np.array([largos[x[fin] == k].mean() if (x[fin] == k).any() else np.nan for k in range(K)])
        occ.append((o / REG.shape[0]).tolist()); rach.append((r / REG.shape[0]).tolist())
    d["ocupacion_regimenes"] = occ
    d["racha_media_regimenes"] = rach
    return d
