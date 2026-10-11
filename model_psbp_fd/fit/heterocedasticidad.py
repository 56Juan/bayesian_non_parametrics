"""
heterocedasticidad.py
=====================
Diagnostico de la corrida 301 (GARCH en los scores), FUERA de las diez metricas:
¿la amplitud de la banda del PSBPM-FD sigue a la varianza condicional verdadera?

El generador C-1 (`pipelines/sim_escenario_GARCH.py`) no tiene regimenes: `internos`
trae `var_condicional` h_tj (T, 4) en la escala unitaria z de cada score activo. En la
escala del score, Var(xi_tj | pasado) = lambda_j h_tj; los scores pasivos son AR(1) de
coeficiente `AR_PASIVO` con varianza condicional constante lambda_j (1 - AR_PASIVO^2).
Dado el pasado completo la ley es gaussiana, de modo que la banda de oraculo al nivel
1 - alpha tiene ancho 2 z_{1-alpha/2} sd_t, por score y por punto de la curva
(sd_t(tau)^2 = sum_k Phi_k(tau)^2 Var(xi_tk | pasado), con los M scores de X^(M)).

El PSBPM-FD condiciona solo en sus rezagos propios, que no contienen h_t (depende de la
historia infinita de innovaciones via beta_1): no se espera que alcance el oraculo. Lo
que se mide es si su banda RESPIRA con h_t: correlacion (Spearman y Pearson) entre su
ancho y el del oraculo, origen a origen, y el coeficiente de variacion de ambos anchos
(cv_psbp / cv_oraculo < 1 es una banda que reacciona menos que la verdad). La banda del
FAR es gaussiana con covarianza fija: su ancho no cambia con t y su correlacion es nula
por construccion, por eso no se calcula.

Los anchos son los de las bandas ya calculadas por `evaluacion_barrido.predecir_barrido`:
`li_s`/`ls_s` por score y `li_f`/`ls_f` por curva (ancho integrado con la cuadratura
trapezoidal comun). No se vuelve a muestrear nada.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

from ..pipelines.sim_scores_comun import AR_PASIVO
from ..utils.quadrature import pesos_trapezoidales

__all__ = ["varianza_condicional_scores", "banda_contra_varianza_condicional"]


def varianza_condicional_scores(esc: Dict) -> np.ndarray:
    """Var(xi_tj | pasado) del generador GARCH, (T, J) en la escala del score."""
    h = esc["interno_var_condicional"][0]
    lam = esc["interno_lambda"]
    v = np.tile(lam * (1.0 - AR_PASIVO ** 2), (h.shape[0], 1))
    v[:, :h.shape[1]] = lam[:h.shape[1]] * h
    return v


def _fila(M, objeto, bloque, ancho, ancho_or):
    return {"M": M, "objeto": objeto, "bloque": bloque, "n": int(ancho.size),
            "spearman": float(spearmanr(ancho, ancho_or)[0]),
            "pearson": float(pearsonr(ancho, ancho_or)[0]),
            "ancho_psbp": float(ancho.mean()), "ancho_oraculo": float(ancho_or.mean()),
            "razon_ancho": float(ancho.mean() / ancho_or.mean()),
            "cv_psbp": float(ancho.std() / ancho.mean()),
            "cv_oraculo": float(ancho_or.std() / ancho_or.mean())}


def banda_contra_varianza_condicional(EST: Dict, M_OK: Sequence[int], DIS: Dict, ORIG: Dict,
                                      esc: Dict, cos_min: float = 0.999) -> pd.DataFrame:
    """
    Por M y bloque (train/test): ancho de la banda del PSBPM-FD contra el del oraculo,
    para la curva (`objeto = "curva"`) y para cada score activo (`"xi_<j>"`). Guarda en
    `EST[M]` los anchos por origen (`ancho_banda_psbp`, `ancho_banda_oraculo`) para la
    figura.
    """
    v = varianza_condicional_scores(esc)
    n_act = esc["interno_var_condicional"].shape[-1]
    Phi, t = esc["interno_Phi"], ORIG["t_orig"]
    assert np.allclose(esc["grilla"], DIS["grilla"]), "grilla del generador != grilla del diseno."
    w = pesos_trapezoidales(DIS["grilla"])
    z = norm.ppf(0.5 + DIS["NIVEL"] / 2)
    filas = []
    for M in M_OK:
        e = EST[M]
        ci = np.asarray(e["component_idx"])
        cos = np.abs(np.diag(e["Psi_grid"].T @ (w[:, None] * Phi[:, ci])))
        assert cos.min() > cos_min, f"[M={M}] Psi_grid no es la base del generador: |cos| = {cos}"
        vt = v[t - 1][:, ci]                                         # t_orig es base-1
        ancho_f = ((e["ls_f"] - e["li_f"]) * w).sum(1)
        ancho_f_or = (2 * z * np.sqrt(vt @ (Phi[:, ci] ** 2).T) * w).sum(1)
        e["ancho_banda_psbp"], e["ancho_banda_oraculo"] = ancho_f, ancho_f_or
        for bloque, m in (("train", ORIG["es_train"]), ("test", ~ORIG["es_train"])):
            filas.append(_fila(M, "curva", bloque, ancho_f[m], ancho_f_or[m]))
            for k in np.flatnonzero(ci < n_act):
                filas.append(_fila(M, f"xi_{ci[k] + 1}", bloque,
                                   (e["ls_s"][:, k] - e["li_s"][:, k])[m],
                                   2 * z * np.sqrt(vt[m, k])))
    return pd.DataFrame(filas)
