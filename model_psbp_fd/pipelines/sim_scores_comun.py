"""
sim_scores_comun.py
===================
Piezas compartidas por los escenarios que especifican la dinamica DIRECTAMENTE
sobre los scores (anexo, seccion de scores): GARCH (corrida 201) y TAR de cuatro
regimenes (202). La multimodalidad (203) usa el motor de `sim_escenario_C1`.

Esquema comun
-------------
    X_t(tau) = mu(tau) + sum_{j<=J} xi_tj phi_j(tau),    J = 10,

con la base de Fourier ortonormal y mu(tau) = sin(2 pi tau), que cae en el espacio
de la base: la representacion es EXACTA y no hay ruido de medicion (sigma_obs = 0).
El score es xi_tj = sqrt(lambda_j) z_tj con lambda_j = lambda_1 rho^(j-1) y z_tj de
varianza marginal unitaria. Los J_a = 4 primeros son ACTIVOS (llevan la dinamica
distintiva) y los demas son AR(1) pasivos de coeficiente 0.3.

`generar_desde_scores` arma la salida a partir de un simulador de z; devuelve en
`internos` la base, los scores, los coeficientes `theta = mu_theta + xi` (lo que
consume la representacion de `_01`), la media condicional exacta (oraculo) y lo
que cada escenario declare como `extras`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..utils.quadrature import pesos_trapezoidales
from .sim_comun import (
    SalidaSimulacion, aplicar_ruido_observacion, diagnostico_comun,
    evaluar_media, grilla_regular, semillas_replicas,
)
from .sim_escenario_C import base_fourier, gram_base
from .sim_escenario_C1 import espectro_nominal, _r2_lineal

__all__ = ["J_SCORES", "BLOQUE_ACTIVO", "AR_PASIVO", "generar_desde_scores", "resumen_scores"]

J_SCORES = 10
BLOQUE_ACTIVO = 4
AR_PASIVO = 0.3


def generar_desde_scores(cfg, simular: Callable, nombre: str, parametros: Optional[dict] = None,
                         soporte: Optional[np.ndarray] = None) -> SalidaSimulacion:
    """
    `simular(T, burn_in, rng)` -> dict con `z` (T, J) en escala unitaria, `oraculo` (T, J)
    (media condicional de z dado el pasado, en la misma escala) y, opcionalmente, arreglos
    `(T, *)` adicionales que se apilan por replica en `internos`.
    """
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    mu = evaluar_media(cfg.media_fn, tau)
    Phi = base_fourier(tau, int(cfg.J))
    lam = espectro_nominal(cfg.J, cfg.lambda_1, cfg.rho)
    s = np.sqrt(lam)
    # Coeficientes de mu en la base (exactos porque mu cae en su espacio).
    mu_theta = np.linalg.solve(gram_base(Phi, tau), Phi.T @ (pesos_trapezoidales(tau) * mu))

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)
    R_, T, G, J = int(cfg.R), int(cfg.T), int(cfg.L), int(cfg.J)
    obs = np.empty((R_, T, G)); curvas = np.empty((R_, T, G)); mc_curva = np.empty((R_, T, G))
    scores = np.empty((R_, T, J)); mc = np.empty((R_, T, J))
    extras: Dict[str, list] = {}
    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        sim = simular(T, int(cfg.burn_in), rng)
        scores[r] = sim["z"] * s
        mc[r] = sim["oraculo"] * s
        for k, v in sim.items():
            if k not in ("z", "oraculo"):
                extras.setdefault(k, []).append(v)
        curvas[r] = mu[None, :] + scores[r] @ Phi.T
        mc_curva[r] = mu[None, :] + mc[r] @ Phi.T
        obs[r] = aplicar_ruido_observacion(curvas[r], cfg.sigma_obs, rng)

    L = int(cfg.n_rezagos)
    if soporte is None:                     # media propia de orden 1
        soporte = np.zeros((J, L, J), dtype=bool)
        for j in range(J):
            soporte[j, 0, j] = True
    internos = {
        "Phi": Phi, "lambda": lam, "escala_scores": s, "scores": scores,
        "mu_theta": mu_theta, "theta": mu_theta[None, None, :] + scores,
        "media_condicional": mc, "media_condicional_curva": mc_curva,
        "soporte": soporte, "n_rezagos": np.array(L), "escenario": np.array(nombre),
        "parametros": parametros if parametros is not None else {},
        **{k: np.stack(v) for k, v in extras.items()},
    }
    salida = SalidaSimulacion(observaciones=obs, curvas=curvas, grilla=tau, media=mu,
                              semillas=registro, config=cfg, internos=internos)
    salida.diagnostico = resumen_scores(salida)
    return salida


def _acf1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else float("nan")


def resumen_scores(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad promediado sobre replicas: R^2 del oraculo, del AR propio de orden
    L y del MCO sobre los L rezagos de las J componentes (por componente activa y agregado
    en L^2), brecha no lineal, heterocedasticidad del residuo del oraculo (cuartil alto de
    |xi_{t-1}| sobre el bajo) y autocorrelacion de sus cuadrados, espectro realizado,
    alineacion con los ejes y ortonormalidad de la base.
    """
    I, cfg = salida.internos, salida.config
    A, L = BLOQUE_ACTIVO, int(I["n_rezagos"])
    XI, MC = I["scores"], I["media_condicional"]
    r2o, r2p, r2l, het, acf2, ev_all, cos_all = [], [], [], [], [], [], []
    r2o_l2, r2l_l2 = [], []
    for r in range(XI.shape[0]):
        xi = XI[r]
        y = xi[L:]
        sst = ((y - y.mean(0)) ** 2).sum(0)
        res = y - MC[r, L:]
        r2o.append(1 - (res ** 2).sum(0) / sst)
        prop = []
        for j in range(xi.shape[1]):
            X = np.column_stack([np.ones(len(y))] + [xi[L - l:len(xi) - l, j] for l in range(1, L + 1)])
            b, *_ = np.linalg.lstsq(X, y[:, j], rcond=None)
            prop.append(1 - ((y[:, j] - X @ b) ** 2).sum() / sst[j])
        r2p.append(prop)
        r2l_r, sse_l = _r2_lineal(xi, L)
        r2l.append(r2l_r)
        r2o_l2.append(1 - (res ** 2).sum() / sst.sum())
        r2l_l2.append(1 - sse_l.sum() / sst.sum())
        h_r, a_r = [], []
        for j in range(A):
            q = np.quantile(np.abs(xi[L - 1:-1, j]), [0.25, 0.75])
            r2j = res[:, j] ** 2
            h_r.append(r2j[np.abs(xi[L - 1:-1, j]) >= q[1]].mean() / r2j[np.abs(xi[L - 1:-1, j]) <= q[0]].mean())
            a_r.append(_acf1(r2j))
        het.append(h_r); acf2.append(a_r)
        ev, U = np.linalg.eigh(np.cov(xi.T))
        o = np.argsort(ev)[::-1]
        ev_all.append(ev[o]); cos_all.append(np.abs(np.diag(U[:, o])))
    ev = np.mean(ev_all, axis=0)
    G = gram_base(I["Phi"], salida.grilla)
    r2o_m, r2p_m = np.mean(r2o, axis=0), np.mean(r2p, axis=0)
    return {
        **diagnostico_comun(salida),
        "n_rezagos": L,
        "r2_oraculo_por_componente": r2o_m[:A + 1].tolist(),
        "r2_ar_propio_por_componente": r2p_m[:A + 1].tolist(),
        "r2_lineal_por_componente": np.mean(r2l, axis=0)[:A + 1].tolist(),
        "brecha_oraculo_ar_propio": (r2o_m - r2p_m)[:A].tolist(),
        "r2_oraculo": float(np.mean(r2o_l2)),
        "r2_lineal": float(np.mean(r2l_l2)),
        "brecha_no_lineal": float(np.mean(r2o_l2) - np.mean(r2l_l2)),
        "heterocedasticidad_q4_q1": np.mean(het, axis=0).tolist(),
        "acf1_cuadrados_residuo": np.mean(acf2, axis=0).tolist(),
        "var_scores": XI.reshape(-1, XI.shape[-1]).var(0).tolist(),
        "lambda_nominal": I["lambda"].tolist(),
        "var_acum_scores": (np.cumsum(ev) / ev.sum()).tolist(),
        "cos_ejes_scores": np.mean(cos_all, axis=0).tolist(),
        "max_abs_scores_activo": float(np.abs(XI[..., :A]).max()),
        "error_ortonormalidad": float(np.abs(G - np.eye(G.shape[0])).max()),
        "T0_referencia": int(np.floor(getattr(cfg, "prop_train_referencia", 0.7) * cfg.T)),
    }
