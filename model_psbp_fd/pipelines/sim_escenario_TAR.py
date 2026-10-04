"""
sim_escenario_TAR.py
=====================
Escenario TAR (corrida 114): cambio de regimen por UMBRAL SUAVE sobre el
REZAGO PROPIO de cada score. No es un algoritmo del anexo; se construyo para
poner a prueba el gating del PSBPM-FD en el diseno de rezago propio, despues de
medir en la 113 que en C-1 el regimen depende de scores ajenos y el rezago
propio no lo ve (salvo en xi_1).

Modelo generador
----------------
    X_t(tau) = mu(tau) + sum_{j=1}^{J} xi_tj phi_j(tau),      J = 10,

con la base de Fourier y mu(tau) = sin(2 pi tau) de la seccion C. Cada score
activo j = 1..4 sigue, en una coordenada cruda u_j, un AR(1) con DOS regimenes
cuya asignacion depende solo de u_{j,t-1}:

    P(R_tj = B | u_{j,t-1}) = Phi((u_{j,t-1} - c_j) / h_j),
    u_tj = mu_A + phi_A u_{j,t-1} + s_A e_tj      si R_tj = A,
    u_tj = mu_B + phi_B u_{j,t-1} + s_B e_tj      si R_tj = B.

A es persistente y tranquilo (phi_A > 0, s_A chico, nivel bajo); B revierte y
es volatil (phi_B < 0, s_B grande, nivel alto). La transicion es un PROBIT
sobre el rezago propio: la misma forma funcional del gating del PSBP, de modo
que el modelo propuesto esta bien especificado en la asignacion y el FAR no.
Las componentes son independientes entre si (regimenes y ruidos propios): la
correlacion contemporanea es ~0 y la FPCA puede alinearse con las phi_j.

El score es z_tj = (u_tj - m_j) / d_j, con (m_j, d_j) la media y la sd
POBLACIONALES de u_j (`MOMENTOS_TAR`, de una simulacion de 200 000 pasos con
seed 7; `momentos_poblacionales` los recalcula), y xi_tj = s_j z_tj con
s_j = sqrt(lambda_j), lambda_j = 0.5 * 0.55^(j-1) como en C. Las componentes
5..10 son AR(1) propios con coeficiente 0.3 y varianza unitaria en z.

Representacion: SIN ruido de medicion (sigma_obs = 0) y sin base estimada. Los
scores son el insumo de los modelos y la base de Fourier del generador, que es
ortonormal en L^2, es la representacion: los coeficientes de la curva en esa
base son theta_t = mu_theta + xi_t (todos los J scores), con mu_theta la
proyeccion de mu (mu = sin(2 pi tau) = phi_1 / sqrt(2), de modo que cae en la
base). `internos["theta"]` y `["mu_theta"]` los entregan.

La dinamica usa UN rezago; el pipeline usa N_LAGS = 2, asi que el rezago 2 es
una covariable sin informacion y sirve para medir la seleccion (PIP).

Calibracion (`PARAMETROS_TAR`, filas j = 1..4)
------------------------------------------------
Version EXAGERADA (pedido del usuario, 2026-10-01): A con equilibrio cerca
del umbral y ruido chico, B lejos (mu_B = 2-2.5) y con reversion fuerte
(phi_B = -0.9), umbral nitido (h = 0.1) en j = 1, 2 y mas suave (h = 0.25) en
j = 3, 4. Se eligio en una grilla de 384 combinaciones (40 000 pasos) entre las
de ocupacion de B en [0.2, 0.6] y racha en [2.5, 15]. Realizacion de la
corrida (seed 41232, T = 1000):

    j   (mu_A, phi_A, s_A, mu_B, phi_B, s_B, c, h)     ocup.B racha R2 oraculo R2 AR(2) brecha var_entre het
    1   (0, 0.8, 0.25, 2.5, -0.9, 1.0, 0, 0.10)        0.42   6.6    0.59       0.18     0.41    0.11    18
    2   (0, 0.6, 0.25, 2.0, -0.9, 1.0, 0, 0.10)        0.46   4.4    0.53       0.02     0.51    0.10    17
    3   (0, 0.8, 0.40, 2.0, -0.9, 1.0, 0, 0.25)        0.41   5.0    0.55       0.13     0.42    0.13     5
    4   (0, 0.6, 0.40, 2.5, -0.9, 1.0, 0, 0.25)        0.52   4.2    0.47       0.03     0.44    0.18     7

"var_entre" = fraccion de la varianza explicada por la incertidumbre de
regimen; "het" = varianza residual del oraculo en el cuartil mas alto de
u_{t-1} sobre la del mas bajo. La media condicional tiene forma de carpa
(sube en A, cae en B) que un AR lineal cruza en diagonal: en j = 2 y 4 el
R^2 lineal es ~0. Un MCO con los rezagos de TODAS las componentes da
0.20 / 0.03 / 0.16 / 0.05: la brecha es no linealidad, no informacion ajena.
Alineacion FPCA: |cos| ejes >= 0.989, corr. contemporanea 0.06, regla del
95 % en M = 5.

El oraculo
----------
Dado u_{j,t-1}, E[u_tj | pasado] = (1 - p) (mu_A + phi_A u) + p (mu_B + phi_B u)
con p = Phi((u - c)/h): exacto, Markov de orden 1 y propio. Se guarda en
`internos["media_condicional"]` (escala xi) y `["media_condicional_curva"]`.
`internos["soporte"]` (J, L, J) marca solo (j, rezago 1, j).

Reproducibilidad: por replica, un `default_rng(hija)`; en cada paso se sortean
J uniformes (regimen) y J normales (innovacion), luego el ruido de observacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.stats import norm

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    aplicar_ruido_observacion,
    diagnostico_comun,
    evaluar_media,
    grilla_regular,
    semillas_replicas,
)
from ..utils.quadrature import pesos_trapezoidales
from .sim_escenario_B1 import media_seno
from .sim_escenario_C import base_fourier, gram_base
from .sim_escenario_C1 import espectro_nominal, _rachas, _r2_lineal

__all__ = [
    "J_SCORES_TAR",
    "BLOQUE_ACTIVO_TAR",
    "PARAMETROS_TAR",
    "MOMENTOS_TAR",
    "ConfigEscenarioTAR",
    "momentos_poblacionales",
    "generar_escenario_TAR",
    "resumen_escenario_TAR",
]

J_SCORES_TAR = 10
BLOQUE_ACTIVO_TAR = 4
AR_PASIVO_TAR = 0.3

# (mu_A, phi_A, s_A, mu_B, phi_B, s_B, c, h) por score activo.
PARAMETROS_TAR = np.array([
    [0.0, 0.8, 0.25, 2.5, -0.9, 1.0, 0.0, 0.10],
    [0.0, 0.6, 0.25, 2.0, -0.9, 1.0, 0.0, 0.10],
    [0.0, 0.8, 0.40, 2.0, -0.9, 1.0, 0.0, 0.25],
    [0.0, 0.6, 0.40, 2.5, -0.9, 1.0, 0.0, 0.25],
])
# (media, sd) poblacionales de u_j: momentos_poblacionales(T=200_000, seed=7).
MOMENTOS_TAR = np.array([
    [+0.1625, 1.2642],
    [+0.2677, 1.1436],
    [-0.0103, 1.2481],
    [+0.5089, 1.3826],
])


@dataclass
class ConfigEscenarioTAR(ConfigObservacion):
    """Mismos defaults de observacion que C (L = 75, T = 1000, R = 1,
    seed = 41232, burn_in = 300), salvo sigma_obs = 0: los scores son el insumo
    y no hay ruido de medicion que filtrar. n_rezagos = 2 es el N_LAGS del
    pipeline; la dinamica verdadera usa 1."""

    L: int = 75
    T: int = 1000
    burn_in: int = 300
    sigma_obs: float = 0.0
    R: int = 1
    seed: int = 41232
    media_fn: Optional[Callable[[np.ndarray], np.ndarray]] = media_seno
    J: int = J_SCORES_TAR
    n_rezagos: int = 2
    lambda_1: float = 0.5
    rho: float = 0.55
    prop_train_referencia: float = 0.70

    def validar(self) -> None:
        super().validar()
        if self.J != J_SCORES_TAR:
            raise ValueError(f"J={self.J}: el escenario fija J = {J_SCORES_TAR}.")
        if self.n_rezagos < 1:
            raise ValueError("n_rezagos >= 1.")
        if self.lambda_1 <= 0 or not 0.0 < self.rho < 1.0:
            raise ValueError("lambda_1 > 0 y rho en (0, 1).")
        if not 0.0 < self.prop_train_referencia < 1.0:
            raise ValueError("prop_train_referencia debe estar en (0, 1).")


def _paso_activo(u_prev, unif, e):
    """Un paso de los scores activos en coordenada u. Devuelve u, regimen, p,
    oraculo y varianza entre regimenes, todos (BLOQUE_ACTIVO_TAR,)."""
    muA, phA, sA, muB, phB, sB, c, h = PARAMETROS_TAR.T
    p = norm.cdf((u_prev - c) / h)
    mA, mB = muA + phA * u_prev, muB + phB * u_prev
    reg = unif < p
    u = np.where(reg, mB + sB * e, mA + sA * e)
    ora = (1 - p) * mA + p * mB
    return u, reg, p, ora, p * (1 - p) * (mA - mB) ** 2


def momentos_poblacionales(T: int = 200_000, seed: int = 7) -> np.ndarray:
    """(media, sd) de u_j en una simulacion larga: la fuente de MOMENTOS_TAR."""
    rng = np.random.default_rng(seed)
    u = np.zeros(BLOQUE_ACTIVO_TAR)
    U = np.empty((T, BLOQUE_ACTIVO_TAR))
    for t in range(T):
        u = _paso_activo(u, rng.random(BLOQUE_ACTIVO_TAR),
                         rng.standard_normal(BLOQUE_ACTIVO_TAR))[0]
        U[t] = u
    U = U[1000:]
    return np.column_stack([U.mean(0), U.std(0)])


def _simular(T: int, burn_in: int, rng: np.random.Generator) -> dict:
    """z (T, J), regimen (T, J), p (T, J), oraculo (T, J) y var_entre (T, J) en z."""
    J, A = J_SCORES_TAR, BLOQUE_ACTIVO_TAR
    m, d = MOMENTOS_TAR[:, 0], MOMENTOS_TAR[:, 1]
    sig_p = np.sqrt(1.0 - AR_PASIVO_TAR ** 2)
    n = int(burn_in) + int(T)
    z = np.zeros((n, J)); reg = np.zeros((n, J), dtype=int)
    P = np.zeros((n, J)); ora = np.zeros((n, J)); ve = np.zeros((n, J))
    u = m.copy()
    zp = np.zeros(J - A)
    for i in range(n):
        unif = rng.random(J)
        e = rng.standard_normal(J)
        u_prev = u
        u, r, p, o, v = _paso_activo(u_prev, unif[:A], e[:A])
        z[i, :A] = (u - m) / d
        reg[i, :A], P[i, :A] = r, p
        ora[i, :A] = (o - m) / d
        ve[i, :A] = v / d ** 2
        ora[i, A:] = AR_PASIVO_TAR * zp
        zp = AR_PASIVO_TAR * zp + sig_p * e[A:]
        z[i, A:] = zp
    c = int(burn_in)
    return {"z": z[c:], "regimen": reg[c:], "p": P[c:], "oraculo": ora[c:],
            "var_entre": ve[c:]}


def _soporte(J: int, L: int) -> np.ndarray:
    s = np.zeros((J, L, J), dtype=bool)
    for j in range(J):
        s[j, 0, j] = True
    return s


def generar_escenario_TAR(cfg: ConfigEscenarioTAR) -> SalidaSimulacion:
    """R replicas del escenario TAR."""
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    mu = evaluar_media(cfg.media_fn, tau)
    Phi = base_fourier(tau, int(cfg.J))
    lam = espectro_nominal(cfg.J, cfg.lambda_1, cfg.rho)
    s = np.sqrt(lam)

    # Coeficientes de la curva en la base del generador (proyeccion L^2, la misma
    # cuadratura del proyecto): theta_t = mu_theta + xi_t, exacto porque mu cae
    # en el span de Phi.
    W = gram_base(Phi, tau)
    mu_theta = np.linalg.solve(W, Phi.T @ (pesos_trapezoidales(tau) * mu))

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)
    R_, T, G, J = int(cfg.R), int(cfg.T), int(cfg.L), int(cfg.J)
    obs = np.empty((R_, T, G)); curvas = np.empty((R_, T, G)); mc_curva = np.empty((R_, T, G))
    scores = np.empty((R_, T, J)); mc = np.empty((R_, T, J)); ventre = np.empty((R_, T, J))
    REG = np.empty((R_, T, J), dtype=int); PB = np.empty((R_, T, J))

    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        sim = _simular(T, int(cfg.burn_in), rng)
        scores[r] = sim["z"] * s
        mc[r] = sim["oraculo"] * s
        ventre[r] = sim["var_entre"] * s ** 2
        REG[r], PB[r] = sim["regimen"], sim["p"]
        curvas[r] = mu[None, :] + scores[r] @ Phi.T
        mc_curva[r] = mu[None, :] + mc[r] @ Phi.T
        obs[r] = aplicar_ruido_observacion(curvas[r], cfg.sigma_obs, rng)

    internos = {
        "Phi": Phi,
        "lambda": lam,
        "escala_scores": s,
        "regimen": REG,
        "p_regimen_B": PB,
        "scores": scores,
        "mu_theta": mu_theta,
        "theta": mu_theta[None, None, :] + scores,
        "media_condicional": mc,
        "media_condicional_curva": mc_curva,
        "varianza_entre_mecanismos": ventre,
        "soporte": _soporte(J, int(cfg.n_rezagos)),
        "n_rezagos": np.array(int(cfg.n_rezagos)),
        "parametros": PARAMETROS_TAR,
        "momentos": MOMENTOS_TAR,
    }
    salida = SalidaSimulacion(observaciones=obs, curvas=curvas, grilla=tau, media=mu,
                              semillas=registro, config=cfg, internos=internos)
    salida.diagnostico = resumen_escenario_TAR(salida)
    return salida


def resumen_escenario_TAR(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad, promediado sobre replicas: ocupacion de B y racha por
    score activo, R^2 del oraculo, del AR(2) PROPIO y del MCO sobre los L
    rezagos de las J componentes, fraccion de varianza entre regimenes,
    heterocedasticidad del oraculo, espectro realizado y alineacion con los ejes.
    """
    cfg, I = salida.config, salida.internos
    A, L = BLOQUE_ACTIVO_TAR, int(I["n_rezagos"])
    XI, MC, VE, REG = I["scores"], I["media_condicional"], I["varianza_entre_mecanismos"], I["regimen"]
    occ, rach, r2o, r2p, r2l, fve, het, ev_all, cos_all, corr = ([] for _ in range(10))
    r2o_l2, r2l_l2 = [], []
    for r in range(XI.shape[0]):
        xi = XI[r]
        occ.append(REG[r, :, :A].mean(0))
        rach.append([np.nanmean(_rachas(REG[r, :, j])[:2]) for j in range(A)])
        y = xi[L:]
        sst = ((y - y.mean(0)) ** 2).sum(0)
        r2o.append(1 - ((y - MC[r, L:]) ** 2).sum(0) / sst)
        prop = []
        for j in range(xi.shape[1]):
            X = np.column_stack([np.ones(len(y))] + [xi[L - l:len(xi) - l, j] for l in range(1, L + 1)])
            b, *_ = np.linalg.lstsq(X, y[:, j], rcond=None)
            prop.append(1 - ((y[:, j] - X @ b) ** 2).sum() / sst[j])
        r2p.append(prop)
        r2l_r, sse_l = _r2_lineal(xi, L)
        r2l.append(r2l_r)
        sse_o = ((y - MC[r, L:]) ** 2).sum(0)
        r2o_l2.append(1 - sse_o.sum() / sst.sum())
        r2l_l2.append(1 - sse_l.sum() / sst.sum())
        fve.append(VE[r].mean(0) / xi.var(0))
        res2 = (y - MC[r, L:]) ** 2
        h_r = []
        for j in range(A):
            q = np.quantile(xi[L - 1:-1, j], [0.25, 0.75])
            h_r.append(res2[xi[L - 1:-1, j] >= q[1], j].mean() / res2[xi[L - 1:-1, j] <= q[0], j].mean())
        het.append(h_r)
        ev, U = np.linalg.eigh(np.cov(xi.T))
        o = np.argsort(ev)[::-1]
        ev_all.append(ev[o]); cos_all.append(np.abs(np.diag(U[:, o])))
        Cr = np.corrcoef(xi[:, :A].T)
        corr.append(float(np.abs(Cr - np.eye(A)).max()))
    ev = np.mean(ev_all, axis=0)
    G = gram_base(I["Phi"], salida.grilla)
    r2o_m, r2p_m = np.mean(r2o, axis=0), np.mean(r2p, axis=0)
    return {
        **diagnostico_comun(salida),
        "n_rezagos": L,
        "ocupacion_regimen_B": np.mean(occ, axis=0).tolist(),
        "racha_media": np.mean(rach, axis=0).tolist(),
        "r2_oraculo_por_componente": r2o_m[:A + 1].tolist(),
        "r2_ar_propio_por_componente": r2p_m[:A + 1].tolist(),
        "r2_lineal_por_componente": np.mean(r2l, axis=0)[:A + 1].tolist(),
        "brecha_oraculo_ar_propio": (r2o_m - r2p_m)[:A].tolist(),
        "r2_oraculo": float(np.mean(r2o_l2)),
        "r2_lineal": float(np.mean(r2l_l2)),
        "brecha_no_lineal": float(np.mean(r2o_l2) - np.mean(r2l_l2)),
        "fraccion_varianza_entre_regimenes": np.mean(fve, axis=0)[:A].tolist(),
        "heterocedasticidad_q4_q1": np.mean(het, axis=0).tolist(),
        "var_scores": XI.reshape(-1, XI.shape[-1]).var(0).tolist(),
        "lambda_nominal": I["lambda"].tolist(),
        "var_acum_scores": (np.cumsum(ev) / ev.sum()).tolist(),
        "cos_ejes_scores": np.mean(cos_all, axis=0).tolist(),
        "corr_contemporanea_max_activo": float(np.mean(corr)),
        "max_abs_scores_activo": float(np.abs(XI[..., :A]).max()),
        "error_ortonormalidad": float(np.abs(G - np.eye(G.shape[0])).max()),
        "T0_referencia": int(np.floor(cfg.prop_train_referencia * cfg.T)),
    }
