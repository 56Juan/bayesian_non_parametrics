r"""
far_kn.py — FAR(p) con la MISMA proporción de varianza que el PSBPM-FD (experimento)
====================================================================================

El `_05` de 107-109 ajusta UN solo FAR(p) para todo el barrido, con `kn` elegido
por `seleccionar_kn` (hold-out sobre train, criterio L2, tope 12 <= K). El
PSBPM-FD del punto `M` en cambio sólo ve las `M` primeras autofunciones. Si
`kn != M` los dos modelos trabajan con subespacios distintos, y la comparación
mezcla "qué modelo es mejor" con "cuántas direcciones de la curva ve cada uno".

Igualar la varianza es fijar `kn = M`, y no por aproximación. `FARp` centra
con la media de train y diagonaliza la covarianza de train de las coordenadas
blanqueadas `theta_w = theta L` (`W = L L^T`, pesos "conteo"). Ése es el
mismo problema generalizado `C u = lambda W u` que resuelve `FPCA_L2` sobre
el mismo bloque. Por eso:

    span(FARp(kn=M).base_)  ==  span(autofunciones FPCA 1..M)   (cosenos = 1)
    valores_propios_ de FARp ==  evals de FPCA                   (misma varianza)

`verificar_subespacio` lo comprueba con `assert` contra los artefactos FPCA de
cada punto. La predicción del FAR(kn=M) vive entonces en `mu + span(psi_1..psi_M)`:
el mismo piso de representación que el PSBPM-FD del punto `M` (en el `_05`,
el `(truncamiento FPCA)`).

Nada de este módulo escribe artefactos: sólo calcula.
"""

from __future__ import annotations

import numpy as np

from model_psbp_fd.fit.far_operador import FARp, seleccionar_kn
from model_psbp_fd.fit import residuos_para_banda, banda_predictiva_modelo
from model_psbp_fd.utils.linalg import safe_chol

__all__ = ["coords_blanqueadas", "ajustar_far", "kn_por_cv", "proyeccion_M",
           "verificar_subespacio", "banda_far", "var_acumulada"]


def coords_blanqueadas(fr, THETA):
    """(THETA_W, L) con W = L L^T la Gram de la base: euclídeo == L^2."""
    W = np.asarray(fr.gram_, dtype=float)
    L = safe_chol(W)
    THETA = np.asarray(THETA, dtype=float)
    THETA_W = THETA @ L
    err = np.max(np.abs(np.einsum("tk,kl,tl->t", THETA, W, THETA) - (THETA_W ** 2).sum(1)))
    assert err / np.max((THETA_W ** 2).sum(1)) < 1e-8, "el blanqueo no preserva L^2"
    return THETA_W, L


def kn_por_cv(THETA_W, T0, p, kn_max=12, criterio="L2"):
    """El kn del `_05`: `seleccionar_kn` sobre train, tope min(kn_max, K)."""
    cv = seleccionar_kn(THETA_W[:T0], kn_max=int(min(kn_max, THETA_W.shape[1])),
                        pesos="conteo", p=p)
    return {"L1": cv.kn_L1, "L2": cv.kn_L2, "Linf": cv.kn_Linf}[criterio], cv


def ajustar_far(fr, THETA_W, L, T0, p, kn):
    """FARp(p, kn) sobre train; devuelve (modelo, X_pred (n_orig, G) con K completo)."""
    far = FARp(p=p, kn=int(kn), pesos="conteo").fit(THETA_W[:T0])
    TW_pred = far.predict_serie(THETA_W)                     # (T - p, K) blanqueado
    TH_pred = np.linalg.solve(L.T, TW_pred.T).T              # vuelta a coeficientes
    return far, fr.reconstruct(TH_pred)


def proyeccion_M(fr, THETA_W, L, T0, M, p):
    """Piso de representación del punto M: mu + P_M (x - mu), en los orígenes."""
    far = FARp(p=1, kn=int(M), pesos="conteo").fit(THETA_W[:T0])
    V = far.base_                                            # ortonormal (pesos conteo)
    mu = far.media_
    TW = mu + (THETA_W - mu) @ V @ V.T
    return fr.reconstruct(np.linalg.solve(L.T, TW.T).T)[p:]


def var_acumulada(THETA_W, T0):
    """Fracción de varianza de train acumulada por las direcciones 1..K."""
    far = FARp(p=1, kn=1, pesos="conteo").fit(THETA_W[:T0])
    ev = np.clip(far.valores_propios_, 0, None)
    return np.cumsum(ev) / ev.sum()


def verificar_subespacio(far, fpca, L, tol=1e-8):
    """Cosenos principales entre FARp.base_ y las M autofunciones FPCA (deben ser 1)."""
    M = fpca.B.shape[1]
    assert far.kn == M, (far.kn, M)
    U = L.T @ np.asarray(fpca.B, float)                      # autofunciones en coords blanqueadas
    U = U / np.linalg.norm(U, axis=0)
    V = far.base_ / np.linalg.norm(far.base_, axis=0)
    cos = np.linalg.svd(V.T @ U, compute_uv=False)
    # FARp divide la covarianza por n y FPCA_L2 por n - 1: se comparan las
    # PROPORCIONES de varianza, que es lo que se quiere igualar.
    ev_far = np.clip(far.valores_propios_, 0, None)
    ev_fp = np.asarray(fpca.evals, float)
    assert np.all(cos > 1 - tol), f"subespacios distintos: cosenos {cos}"
    assert np.allclose(ev_far[:M] / ev_far.sum(), ev_fp[:M] / ev_fp.sum(), rtol=1e-6), (ev_far, ev_fp)
    return cos


def banda_far(X_obj, X_far, es_train, kn, nivel, por_tau=True, correccion=True):
    """La banda del `_05`: residuos de train, sigma(tau), inflada por sqrt(1 + kn/n_train)."""
    n_train = int(np.sum(es_train))
    corr = np.sqrt(1.0 + kn / n_train) if correccion else None
    r = residuos_para_banda(X_obj, X_far, es_train)
    return banda_predictiva_modelo(X_far, r, nivel=nivel, por_tau=por_tau,
                                   correccion_estimacion=corr)
