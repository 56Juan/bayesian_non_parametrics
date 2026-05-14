"""
PSBP Predictor — Inferencia post-MCMC para Probit Stick-Breaking Process
=========================================================================

Consume las trazas producidas por `PSBPSampler` y entrega:

    1. Predicción puntual E[y|x] (media posterior).
    2. Densidad condicional f(y|x) sobre una grilla de y.
    3. Intervalos de credibilidad sobre E[y|x].
    4. Probabilidades de inclusión global por variable.
    5. Métrica RMSE.

CAMBIOS v2 (D3 — opción A):
---------------------------
La estandarización es **totalmente externa**. El predictor opera por
completo en la escala en la que fueron entrenadas las trazas: si el
sampler se entrenó con y estandarizado, predict/predict_density/
predict_interval/rmse devuelven valores en escala estandarizada. Si se
entrenó con y crudo, devuelven valores en escala cruda. La clase no
desestandariza nada.

Esto elimina la inconsistencia MATLAB (estandarizar con ddof=0,
desestandarizar con ddof=1) y simplifica el contrato.

Author: model_psbp_fd
"""

from __future__ import annotations
from typing import Dict, Tuple, Union
import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class PSBPPredictor:
    """
    Inferencia posterior a partir de las trazas MCMC del PSBPSampler.

    Parámetros
    ----------
    traces : dict
        Diccionario de trazas devuelto por `PSBPSampler.fit().traces`.
        Claves requeridas: betajhout, beta0hout, tauhout, alphahout,
        psijhout, Gammajhout, osumout.

    burn : int
        Número de iteraciones a descartar como burn-in.

    Atributos
    ---------
    n_post_ : int
        Número de muestras post-burn-in.

    n_components_ : int
        Truncamiento N del proceso.

    n_features_ : int
        Número de covariables p (sin intercepto).

    Notas
    -----
    El predictor es **agnóstico a la escala**: trabaja en la misma escala
    en la que se entrenaron las trazas. Las salidas de `predict`,
    `predict_density`, `predict_interval` y `rmse` están todas en la
    escala en que el usuario entrenó el sampler. Si el sampler usó
    datos estandarizados, las salidas están en escala estandarizada;
    la desestandarización (si se desea) debe hacerse fuera de la clase.

    El input X de los métodos debe estar en la misma escala que el X
    de entrenamiento y debe incluir columna de intercepto (X[:, 0] == 1).
    """

    def __init__(self,
                 traces: Dict[str, np.ndarray],
                 burn: int):
        self._validate_traces(traces, burn)
        self.burn = int(burn)

        # Slicing post-burn-in (vista, no copia)
        post = slice(self.burn, None)
        self._betajh   = traces["betajhout"][post]    # (T, N, p)
        self._beta0h   = traces["beta0hout"][post]    # (T, N)
        self._tauh     = traces["tauhout"][post]      # (T, N)
        self._alphah   = traces["alphahout"][post]    # (T, N-1)
        self._psijh    = traces["psijhout"][post]     # (T, N-1, p)
        self._Gammajh  = traces["Gammajhout"][post]   # (T, N-1, p)
        self._osumout  = traces["osumout"][post]      # (T, p)

        self.n_post_       = self._betajh.shape[0]
        self.n_components_ = self._betajh.shape[1]
        self.n_features_   = self._betajh.shape[2]

    # ─────────────────────────────────────────────────────────────────────
    # Validación
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_traces(traces: Dict[str, np.ndarray], burn: int) -> None:
        required = {"betajhout", "beta0hout", "tauhout", "alphahout",
                    "psijhout", "Gammajhout", "osumout"}
        missing = required - set(traces.keys())
        if missing:
            raise KeyError(f"traces le faltan claves: {missing}")

        nsim = traces["betajhout"].shape[0]
        if not (0 <= burn < nsim):
            raise ValueError(
                f"burn={burn} fuera de rango [0, {nsim})"
            )

    def _validate_X(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D; recibido shape {X.shape}")
        if X.shape[1] != self.n_features_ + 1:
            raise ValueError(
                f"X tiene {X.shape[1]} columnas; se esperan "
                f"{self.n_features_ + 1} (intercepto + {self.n_features_} feats)"
            )
        if not np.allclose(X[:, 0], 1.0):
            raise ValueError(
                "Se espera que X[:, 0] sea la columna de unos (intercepto)"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Núcleo: cálculo de pesos π_h(x) y media μ_h(x) por iteración
    # ─────────────────────────────────────────────────────────────────────
    def _compute_weights(self,
                         Xnoint: np.ndarray,
                         alphah_t: np.ndarray,
                         psijh_t: np.ndarray,
                         Gammajh_t: np.ndarray) -> np.ndarray:
        """
        Pesos π_h(x) del stick-breaking probit para una iteración t.

        Parámetros
        ----------
        Xnoint    : (n, p)
        alphah_t  : (N-1,)
        psijh_t   : (N-1, p)
        Gammajh_t : (N-1, p)

        Retorna
        -------
        phx : (n, N)
            π_h(x_i) para i = 1..n, h = 1..N.
        """
        n = Xnoint.shape[0]
        N = self.n_components_

        # eta_{h}(x_i) = alpha_h - sum_j psi_{hj} |x_{ij} - Gamma_{hj}|
        # forma: (N-1, n)
        diff = np.abs(Xnoint[None, :, :] - Gammajh_t[:, None, :])  # (N-1, n, p)
        eta  = alphah_t[:, None] - np.sum(psijh_t[:, None, :] * diff, axis=2)
        vhx  = norm.cdf(eta)  # (N-1, n)

        # phx_h = vhx_h * prod_{l<h} (1 - vhx_l)
        # Calculamos cumprod de (1 - vhx) a lo largo de h.
        log_one_minus = np.log(np.clip(1.0 - vhx, 1e-300, 1.0))  # (N-1, n)
        # cumlog[h] = sum_{l<=h} log(1 - vhx_l)
        cumlog = np.cumsum(log_one_minus, axis=0)               # (N-1, n)

        phx = np.zeros((N, n))
        phx[0] = vhx[0]
        if N > 1:
            # h = 1..N-2 (índices 1-based 2..N-1)
            # phx_h = vhx_h * exp(cumlog_{h-1})
            phx[1:N - 1] = vhx[1:N - 1] * np.exp(cumlog[:N - 2])
            # Última componente: prod (1 - vhx_l) sobre todos los l
            phx[N - 1] = np.exp(cumlog[N - 2])

        return phx.T  # (n, N)

    def _compute_mu_h(self,
                      X: np.ndarray,
                      beta0h_t: np.ndarray,
                      betajh_t: np.ndarray) -> np.ndarray:
        """
        Media condicional por componente: μ_h(x_i) = β_{h0} + x_i' β_h.

        Parámetros
        ----------
        X         : (n, p+1)  — con intercepto
        beta0h_t  : (N,)
        betajh_t  : (N, p)

        Retorna
        -------
        mu : (n, N)
        """
        # X[:, 0] = 1, así que X[:, 0] * beta0h_t[h] = beta0h_t[h]
        return X[:, 0:1] * beta0h_t[None, :] + X[:, 1:] @ betajh_t.T

    # ─────────────────────────────────────────────────────────────────────
    # 1. Predicción puntual E[y|x]
    # ─────────────────────────────────────────────────────────────────────
    def predict(self,
                X: np.ndarray,
                return_std: bool = False
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predicción puntual E[y|x] = E_post[Σ_h π_h(x) · μ_h(x)].

        Parámetros
        ----------
        X : ndarray, shape (n, p+1)
            Matriz de diseño con intercepto, **en la misma escala que el
            X de entrenamiento**.
        return_std : bool
            Si True, retorna también la desviación estándar posterior
            de la predicción puntual.

        Retorna
        -------
        y_pred : ndarray, shape (n,)
            Predicción puntual en la escala de entrenamiento.
        y_std  : ndarray, shape (n,) — solo si return_std=True
            Desviación estándar posterior en la escala de entrenamiento.
        """
        self._validate_X(X)
        Xnoint = X[:, 1:]
        n = X.shape[0]
        T = self.n_post_

        E_per_iter = np.zeros((T, n), dtype=np.float64)
        for t in range(T):
            phx  = self._compute_weights(
                Xnoint, self._alphah[t], self._psijh[t], self._Gammajh[t]
            )                                            # (n, N)
            mu_h = self._compute_mu_h(
                X, self._beta0h[t], self._betajh[t]
            )                                            # (n, N)
            E_per_iter[t] = np.sum(phx * mu_h, axis=1)   # (n,)

        y_pred = E_per_iter.mean(axis=0)
        if not return_std:
            return y_pred

        y_std = E_per_iter.std(axis=0, ddof=1)
        return y_pred, y_std

    # ─────────────────────────────────────────────────────────────────────
    # 2. Densidad condicional f(y|x)
    # ─────────────────────────────────────────────────────────────────────
    def predict_density(self,
                        X: np.ndarray,
                        y_grid: np.ndarray,
                        return_per_iter: bool = False
                        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Densidad condicional posterior f(y|x).

        f(y|x) = E_post[ Σ_h π_h(x) · N(y; μ_h(x), 1/√τ_h) ]

        Parámetros
        ----------
        X      : (n, p+1) — diseño con intercepto, escala de entrenamiento
        y_grid : (G,)     — valores de y donde evaluar la densidad,
                            **en la misma escala** que se usó para entrenar
        return_per_iter : bool
            Si True, retorna también la densidad por iteración (T, n, G).

        Retorna
        -------
        density_mean : ndarray, shape (n, G)
            Media posterior de f(y|x).
        density_per_iter : ndarray, shape (T, n, G) — solo si solicitado.
        """
        self._validate_X(X)
        if y_grid.ndim != 1:
            raise ValueError(f"y_grid debe ser 1D; shape={y_grid.shape}")

        Xnoint = X[:, 1:]
        n = X.shape[0]
        G = y_grid.shape[0]
        T = self.n_post_

        if return_per_iter:
            density = np.zeros((T, n, G), dtype=np.float64)

        density_sum = np.zeros((n, G), dtype=np.float64)

        for t in range(T):
            phx     = self._compute_weights(
                Xnoint, self._alphah[t], self._psijh[t], self._Gammajh[t]
            )                                                  # (n, N)
            mu_h    = self._compute_mu_h(
                X, self._beta0h[t], self._betajh[t]
            )                                                  # (n, N)
            sigma_h = 1.0 / np.sqrt(self._tauh[t])             # (N,)

            # f_t(y|x_i) = Σ_h phx[i,h] * N(y_grid; mu_h[i,h], sigma_h[h])
            # broadcast: (n, N, G)
            z      = (y_grid[None, None, :] - mu_h[:, :, None]) / sigma_h[None, :, None]
            comp   = norm.pdf(z) / sigma_h[None, :, None]      # densidad por componente
            f_t    = np.sum(phx[:, :, None] * comp, axis=1)    # (n, G)

            density_sum += f_t
            if return_per_iter:
                density[t] = f_t

        density_mean = density_sum / T

        if return_per_iter:
            return density_mean, density
        return density_mean

    # ─────────────────────────────────────────────────────────────────────
    # 3. Intervalo de credibilidad sobre E[y|x]
    # ─────────────────────────────────────────────────────────────────────
    def predict_interval(self,
                         X: np.ndarray,
                         level: float = 0.95
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Intervalo de credibilidad posterior sobre E[y|x].

        Parámetros
        ----------
        X : (n, p+1) — diseño con intercepto, escala de entrenamiento
        level : float ∈ (0, 1) — nivel de credibilidad (default 0.95)

        Retorna
        -------
        y_pred : (n,) — media posterior, escala de entrenamiento
        y_lo   : (n,) — cuantil inferior, escala de entrenamiento
        y_hi   : (n,) — cuantil superior, escala de entrenamiento
        """
        if not (0.0 < level < 1.0):
            raise ValueError(f"level debe estar en (0, 1); recibido {level}")

        self._validate_X(X)
        Xnoint = X[:, 1:]
        n = X.shape[0]
        T = self.n_post_

        E_per_iter = np.zeros((T, n), dtype=np.float64)
        for t in range(T):
            phx  = self._compute_weights(
                Xnoint, self._alphah[t], self._psijh[t], self._Gammajh[t]
            )
            mu_h = self._compute_mu_h(X, self._beta0h[t], self._betajh[t])
            E_per_iter[t] = np.sum(phx * mu_h, axis=1)

        alpha = (1.0 - level) / 2.0
        y_pred = E_per_iter.mean(axis=0)
        y_lo   = np.quantile(E_per_iter, alpha, axis=0)
        y_hi   = np.quantile(E_per_iter, 1.0 - alpha, axis=0)
        return y_pred, y_lo, y_hi

    # ─────────────────────────────────────────────────────────────────────
    # 4. Probabilidades de inclusión global
    # ─────────────────────────────────────────────────────────────────────
    def inclusion_probs(self) -> np.ndarray:
        """
        P(γ_j = 1 | data) por variable, estimada como
        1 − mean_{t ≥ burn}(osumout[t, j]).

        Nota sobre convención:
        ----------------------
        En MATLAB el código original guarda `gloprob` que es la
        probabilidad de **NO inclusión** (P(todos γ_{·,j}=0 entre los
        clusters activos)). Aquí se devuelve directamente la cantidad
        más usada en la práctica: P(la variable está incluida en al
        menos un cluster activo).

        Retorna
        -------
        incl : (p,)
            P(γ_j = 1 | data) por variable.
        """
        gloprob = self._osumout.mean(axis=0)
        return 1.0 - gloprob

    # ─────────────────────────────────────────────────────────────────────
    # 5. RMSE
    # ─────────────────────────────────────────────────────────────────────
    def rmse(self,
             X: np.ndarray,
             y_obs: np.ndarray) -> float:
        """
        RMSE entre predicción puntual E[y|x] y `y_obs`, **en la escala
        de entrenamiento**.

        Si el usuario quiere RMSE en escala original cuando entrenó con
        datos estandarizados, debe desestandarizar `y_obs` antes de
        pasarlo aquí, o reescalar la salida después.

        Parámetros
        ----------
        X     : (n, p+1) — diseño con intercepto, escala de entrenamiento
        y_obs : (n,)     — respuesta observada **en la misma escala que se
                           usó para entrenar** (esto es responsabilidad
                           del usuario)

        Retorna
        -------
        rmse : float
        """
        if y_obs.ndim != 1:
            raise ValueError(f"y_obs debe ser 1D; shape={y_obs.shape}")
        if y_obs.shape[0] != X.shape[0]:
            raise ValueError(
                f"X.shape[0]={X.shape[0]} != y_obs.shape[0]={y_obs.shape[0]}"
            )

        y_pred = self.predict(X)
        return float(np.sqrt(np.mean((y_pred - y_obs) ** 2)))
