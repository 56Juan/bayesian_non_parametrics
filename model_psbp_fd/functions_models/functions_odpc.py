"""
functions_odpc.py
=================
Componentes principales dinamicas UNILATERALES (ODPC; Pena, Smucler & Yohai
2019, JASA 114(528), "Forecasting Multiple Time Series with One-Sided Dynamic
Principal Components") llevadas a datos funcionales. Corridas 110-112.

Por que unilaterales
--------------------
Las componentes dinamicas de Brillinger / Hormann, Kidzinski & Hallin (2015)
usan un filtro BILATERAL: el score de t combina X_{t-L}, ..., X_{t+L}. Como
covariable para predecir X_{t+1} eso es fuga, aunque el filtro se estime solo
con train: la fuga esta en la formula del score, no en la estimacion. Las
ODPC se DEFINEN con el presente y el pasado, y se ajustan minimizando el error
de reconstruccion:

    f_t   = sum_{h=0}^{k1} a_h^T z_{t-h}                 (componente)
    z_t  ~= sum_{h=0}^{k2} beta_h f_{t-h}                (reconstruccion)

con (a, beta) por minimos cuadrados alternados. La reconstruccion usa la
componente actual y sus rezagos: al pronosticar se predice f_{t+1} y los
rezagos, ya observados, entran con su valor real.

Version funcional
-----------------
z_t son los `K_din` primeros scores de la FPCA ESTATICA en metrica L2
(ortonormales), de modo que el error euclideo de la reconstruccion de los K
scores estaticos es el error L2 de la curva. Los objetos de base (W, B_full,
mu_theta) salen de `FPCA_L2` y se ajustan solo con train.

Varias componentes, secuenciales: la componente m se ajusta sobre el RESIDUO
de reconstruccion de las m-1 anteriores (como en el paper), pero su filtro se
aplica a los scores estaticos originales, no al residuo. Asi el score f_m es
un filtro causal fijo de las curvas, NO depende de M, y el M-esimo punto del
barrido es la suma de las primeras M reconstrucciones.

Curva
-----
    X_t = mu + sum_{h=0}^{k2} f_{t-h} Psi_h^T,     Psi_h = Phi B_full beta_h^T.

`reconstruct(F)` es la parte CONTEMPORANEA (h = 0), la unica que depende del
score que se predice; `desplazamiento(F, desde)` es la parte h >= 1, que usa
los scores reales ya observados y actua como una media que cambia con el
origen. `reconstruct_serie(F)` es la suma, sobre una serie continua.

Sin intercepto: z y el residuo estan centrados con la media de train.

Retencion temporal
------------------
`fit` solo ve train. `transform` necesita la serie CONTINUA (la fila t usa las
curvas t-k1..t); antes de la primera curva se supone score estatico cero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..utils.quadrature import pesos_trapezoidales
from .functions_fpca import FPCA_L2

__all__ = ["ODPC_Funcional", "filtrar_causal", "rezagar"]


def rezagar(F: np.ndarray, h: int) -> np.ndarray:
    """F desplazado h filas hacia abajo, con ceros al inicio."""
    F = np.atleast_2d(F)
    if h == 0:
        return F
    out = np.zeros_like(F)
    if h < F.shape[0]:
        out[h:] = F[:-h]
    return out


def filtrar_causal(S: np.ndarray, filtros: np.ndarray) -> np.ndarray:
    """
    f_t = sum_j filtros[j]^T s_{t-j}, con scores cero antes de la primera fila.

    S       : (T, Kd)       scores estaticos centrados.
    filtros : (J, Kd, M)    filtros[j] pesa la curva t-j.
    """
    S = np.atleast_2d(np.asarray(S, dtype=float))
    F = np.zeros((S.shape[0], filtros.shape[2]))
    for j in range(filtros.shape[0]):
        F += rezagar(S, j) @ filtros[j]
    return F


def _ajustar_componente(Zlag, R, k1, k2, max_iter, tol):
    """
    Una ODPC por minimos cuadrados alternados.

    Zlag : (n, P) [z_t, z_{t-1}, ..., z_{t-k1}], P = (k1+1) Kd.
    R    : (n, K) objetivo (residuo) a reconstruir.
    Solo se usan las filas t >= k1 + k2, donde ningun f_{t-h} toca el relleno.
    """
    n, P = Zlag.shape
    t0 = k1 + k2
    Rv = R[t0:]
    lagZ = [rezagar(Zlag, h)[t0:] for h in range(k2 + 1)]      # Zlag_{t-h}
    ZZ = {(h, g): lagZ[h].T @ lagZ[g] for h in range(k2 + 1) for g in range(h, k2 + 1)}
    ZR = [lagZ[h].T @ Rv for h in range(k2 + 1)]                # (P, K)

    # Inicio: filtro estatico (solo h = 0) en la direccion de maxima
    # covarianza con el objetivo.
    U, _, _ = np.linalg.svd(lagZ[0].T @ Rv, full_matrices=False)
    a = U[:, 0].copy()
    mse_prev, beta = np.inf, None
    ridge = 1e-10 * np.trace(ZZ[(0, 0)]) / P
    for it in range(max_iter):
        a /= np.linalg.norm(a)
        F = np.column_stack([lz @ a for lz in lagZ])            # (nv, k2+1)
        beta, *_ = np.linalg.lstsq(F, Rv, rcond=None)           # (k2+1, K)
        mse = float(((Rv - F @ beta) ** 2).sum())
        if mse_prev - mse <= tol * max(mse, 1e-300):
            break
        mse_prev = mse
        # a | beta: problema cuadratico en a.
        BB = beta @ beta.T                                      # (k2+1, k2+1)
        A = np.zeros((P, P))
        for (h, g), M_hg in ZZ.items():
            A += BB[h, g] * (M_hg if h == g else M_hg + M_hg.T)
        b = sum(ZR[h] @ beta[h] for h in range(k2 + 1))
        a = np.linalg.solve(A + ridge * np.eye(P), b)
    a /= np.linalg.norm(a)
    F = np.column_stack([lz @ a for lz in lagZ])
    beta, *_ = np.linalg.lstsq(F, Rv, rcond=None)
    # Signo: la mayor carga contemporanea positiva.
    s = np.sign(beta[0][np.argmax(np.abs(beta[0]))]) or 1.0
    return a * s, beta * s, it + 1


@dataclass
class ODPC_Funcional(FPCA_L2):
    """
    ODPC sobre los scores de la FPCA estatica, con la interfaz de `FPCA_L2`.

    Parametros
    ----------
    k1       : rezagos del filtro (la componente usa las curvas t-k1..t).
    k2       : rezagos de la reconstruccion (usa f_t..f_{t-k2}).
    K_din    : scores estaticos que entran al filtro.
    M_max    : componentes que se ajustan; set_M elige las primeras M.
    max_iter, tol : de los minimos cuadrados alternados.

    Atributos tras `fit` (ademas de los de `FPCA_L2`)
    -------------------------------------------------
    filtros_full : (k1+1, K_din, M_max)   a_{m,h}
    betas_full   : (M_max, k2+1, K)       beta_{m,h} sobre los K scores estaticos
    evals, var_ratio, var_cum : varianza que recupera cada componente en train
        (incremental); las estaticas quedan en `*_estatica`.
    """

    k1: int = 1
    k2: int = 1
    K_din: int = 10
    M_max: int = 10
    max_iter: int = 3000
    tol: float = 1e-10

    evals_estatica: np.ndarray = field(default=None, repr=False)
    var_ratio_estatica: np.ndarray = field(default=None, repr=False)
    var_cum_estatica: np.ndarray = field(default=None, repr=False)
    filtros_full: np.ndarray = field(default=None, repr=False)
    betas_full: np.ndarray = field(default=None, repr=False)
    iteraciones: list = field(default=None, repr=False)
    lambdas_train: np.ndarray = field(default=None, repr=False)
    _S_tr: np.ndarray = field(default=None, repr=False)
    _F_tr: np.ndarray = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # AJUSTE
    # ------------------------------------------------------------------
    def fit(self, THETA_train: np.ndarray, Phi: np.ndarray,
            tau: np.ndarray) -> "ODPC_Funcional":
        super().fit(THETA_train, Phi, tau)          # base estatica, solo train
        THETA_train = np.atleast_2d(np.asarray(THETA_train, dtype=float))
        n, K = THETA_train.shape
        k1, k2 = int(self.k1), int(self.k2)
        if min(k1, k2) < 0:
            raise ValueError("k1 y k2 deben ser >= 0.")
        Kd = int(min(self.K_din, K))
        Mx = int(min(self.M_max, Kd * (k1 + 1)))

        S_full = (THETA_train - self.mu_theta) @ (self.W @ self.B_full)   # (n, K)
        Z = S_full[:, :Kd]
        Zlag = np.hstack([rezagar(Z, h) for h in range(k1 + 1)])       # (n, P)

        filtros = np.zeros((k1 + 1, Kd, Mx))
        betas = np.zeros((Mx, k2 + 1, K))
        F = np.zeros((n, Mx))
        R = S_full.copy()
        self.iteraciones = []
        for m in range(Mx):
            a, beta, it = _ajustar_componente(Zlag, R, k1, k2,
                                              self.max_iter, self.tol)
            filtros[:, :, m] = a.reshape(k1 + 1, Kd)
            betas[m] = beta
            F[:, m] = Zlag @ a
            R = R - sum(rezagar(F[:, [m]], h) * beta[h] for h in range(k2 + 1))
            self.iteraciones.append(it)
        self.filtros_full, self.betas_full = filtros, betas
        self._S_tr, self._F_tr = S_full, F

        self.evals_estatica = self.evals.copy()
        self.var_ratio_estatica = self.var_ratio.copy()
        self.var_cum_estatica = self.var_cum.copy()
        acum = np.array([self.var_reconstruida(m) for m in range(1, Mx + 1)])
        self.var_cum = acum
        self.var_ratio = np.diff(np.r_[0.0, acum])
        self.evals = self.var_ratio * float(self.evals_estatica.sum())
        self.lambdas_train = F[self._burn:].var(axis=0, ddof=1)
        self.n_components = None
        return self

    @property
    def _burn(self) -> int:
        """Filas de train donde algun f_{t-h} usa el relleno inicial."""
        return int(min(self.k1 + self.k2, self._S_tr.shape[0] - 1))

    def var_reconstruida(self, M: Optional[int] = None) -> float:
        """Fraccion de la varianza de train que recuperan las M primeras."""
        M = self.M if M is None else int(M)
        b = self._burn
        S = self._S_tr[b:]
        Shat = self._recon_estatica(self._F_tr[:, :M], M)[b:]
        return float(1.0 - ((S - Shat) ** 2).sum() / (S ** 2).sum())

    def seleccionar_M(self, umbral: float = 0.99) -> int:
        """Menor M cuya reconstruccion recupera `umbral` en train."""
        self._check_fitted()
        idx = np.searchsorted(self.var_cum, umbral)
        return int(min(idx + 1, self.var_cum.size))

    def set_M(self, M: int) -> "ODPC_Funcional":
        self._check_fitted()
        Mx = self.filtros_full.shape[2]
        if not (isinstance(M, (int, np.integer)) and 1 <= M <= Mx):
            raise ValueError(f"M debe ser entero en [1, {Mx}]; recibido {M!r}.")
        self.n_components = int(M)
        return self

    # ------------------------------------------------------------------
    # MAPAS
    # ------------------------------------------------------------------
    def _recon_estatica(self, F: np.ndarray, M: int) -> np.ndarray:
        """(T, K) scores estaticos reconstruidos con las M primeras."""
        F = np.atleast_2d(F)
        out = np.zeros((F.shape[0], self.betas_full.shape[2]))
        for h in range(self.betas_full.shape[1]):
            out += rezagar(F[:, :M], h) @ self.betas_full[:M, h, :]
        return out

    def B_rezago(self, h: int) -> np.ndarray:
        """(K, M) coeficientes de base del rezago h: B_full beta_h^T."""
        self._check_M()
        return self.B_full @ self.betas_full[:self.M, h, :].T

    @property
    def B(self) -> np.ndarray:
        """(K, M) parte contemporanea (h = 0)."""
        return self.B_rezago(0)

    @property
    def filtros(self) -> np.ndarray:
        return self.filtros_full[:, :, :self.M]

    @property
    def lambdas(self) -> np.ndarray:
        """(M,) varianza de las componentes en train."""
        return self.lambdas_train[:self.M]

    def scores_estaticos(self, THETA: np.ndarray) -> np.ndarray:
        THETA = np.atleast_2d(np.asarray(THETA, dtype=float))
        Kd = self.filtros_full.shape[1]
        return (THETA - self.mu_theta) @ (self.W @ self.B_full[:, :Kd])

    def transform(self, THETA: np.ndarray) -> np.ndarray:
        """Serie CONTINUA de coeficientes (T, K) -> componentes (T, M)."""
        self._check_M()
        return filtrar_causal(self.scores_estaticos(THETA), self.filtros)

    def transform_completo(self, THETA: np.ndarray) -> np.ndarray:
        """Como `transform` con las M_max componentes; no depende de M."""
        self._check_fitted()
        return filtrar_causal(self.scores_estaticos(THETA), self.filtros_full)

    def transform_prediccion(self, THETA: np.ndarray, THETA_pred: np.ndarray,
                             desde: int) -> np.ndarray:
        """
        Componentes con la curva t sustituida por su PREDICCION a un paso,
        t = desde..T-1, y rezagos reales: f_t + a_0^T (s_pred_t - s_t).
        """
        self._check_M()
        THETA = np.atleast_2d(np.asarray(THETA, dtype=float))
        F = self.transform(THETA)[desde:]
        S_real = self.scores_estaticos(THETA)[desde:]
        S_pred = self.scores_estaticos(THETA_pred)
        if S_pred.shape != S_real.shape:
            raise ValueError(f"THETA_pred {S_pred.shape} no alinea con "
                             f"THETA[{desde}:] {S_real.shape}.")
        return F + (S_pred - S_real) @ self.filtros[0]

    def desplazamiento(self, F: np.ndarray, desde: int = 0) -> np.ndarray:
        """
        (T-desde, G) parte de la curva de t que aportan f_{t-1..t-k2}: con la
        serie real F es conocida en el origen y actua como media variable.
        """
        self._check_M()
        F = np.atleast_2d(F)[:, :self.M]
        D = np.zeros((F.shape[0], self.Phi.shape[0]))
        for h in range(1, self.betas_full.shape[1]):
            D += rezagar(F, h) @ (self.Phi @ self.B_rezago(h)).T
        return D[desde:]

    def reconstruct_serie(self, F: np.ndarray) -> np.ndarray:
        """Serie CONTINUA de componentes (T, M) -> curvas (T, G), todos los h."""
        return self.reconstruct(F) + self.desplazamiento(F)

    # ------------------------------------------------------------------
    # VERIFICACION
    # ------------------------------------------------------------------
    def verificar(self, THETA_train=None, fr=None, tol: float = 1e-10) -> dict:
        """Como `FPCA_L2.verificar`; sin M fijado usa M = M_max."""
        self._check_fitted()
        M_prev = self.n_components
        try:
            if M_prev is None:
                self.n_components = int(self.filtros_full.shape[2])
            return self._verificar_impl(THETA_train, fr, tol)
        finally:
            self.n_components = M_prev

    def _verificar_impl(self, THETA_train, fr, tol) -> dict:
        M = self.M
        K = int(self.B_full.shape[0])
        d = {
            "M": M, "K": K, "K_din": int(self.filtros_full.shape[1]),
            "k1": int(self.k1), "k2": int(self.k2),
            "n_ajuste": self.n_ajuste, "cond_W": self.cond_W,
            "var_explicada": float(self.var_cum[M - 1]),
            "iteraciones_als": list(self.iteraciones[:M]),
            "err_ortonormalidad_estatica": float(np.abs(
                self.B_full.T @ self.W @ self.B_full - np.eye(K)).max()),
        }
        S_chk = np.eye(M)
        d["err_rutas_reconstruccion"] = float(np.abs(
            self.reconstruct(S_chk)
            - self.inverse_transform(S_chk) @ self.Phi.T).max())

        if fr is not None:
            rng = np.random.default_rng(0)
            Th = rng.standard_normal((max(3, min(8, K)), K))
            escala = float(np.abs(self.Phi).max()) or 1.0
            d["err_linealidad_reconstruct_rel"] = float(np.abs(
                np.asarray(fr.reconstruct(Th), dtype=float) - Th @ self.Phi.T
            ).max()) / escala

        if THETA_train is not None:
            # La curva reconstruida por la ruta de curvas debe coincidir con la
            # de scores estaticos: comprueba desplazamiento y B_rezago.
            b = self._burn
            TH = np.atleast_2d(THETA_train)
            F = self.transform(TH)
            X1 = self.reconstruct_serie(F)[b:]
            X2 = (self.mu_theta + self._recon_estatica(F, M) @ self.B_full.T) @ self.Phi.T
            escala = float(np.abs(X2[b:]).max()) or 1.0
            d["err_serie_vs_estatica_rel"] = float(np.abs(X1 - X2[b:]).max()) / escala
            w = pesos_trapezoidales(self.tau)
            Xc = (TH - self.mu_theta)[b:] @ self.Phi.T
            d["frac_varianza_no_reconstruida"] = float(
                ((Xc - (X1 - self.mu_grid)) ** 2 * w).sum(1).mean()
                / (Xc ** 2 * w).sum(1).mean())

        criterios = ("err_ortonormalidad_estatica", "err_rutas_reconstruccion",
                     "err_linealidad_reconstruct_rel", "err_serie_vs_estatica_rel")
        d["criterios_evaluados"] = [k for k in criterios if k in d]
        d["todo_ok"] = bool(max(d[k] for k in d["criterios_evaluados"]) < tol)
        return d

    def resumen_componentes(self):
        import pandas as pd
        Mx = self.filtros_full.shape[2]
        return pd.DataFrame({
            "componente": np.arange(1, Mx + 1),
            "var_ratio": self.var_ratio, "var_acum": self.var_cum,
            "var_acum_estatica": self.var_cum_estatica[:Mx],
            "iteraciones_als": self.iteraciones,
        })
