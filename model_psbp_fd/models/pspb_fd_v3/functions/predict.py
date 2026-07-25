r"""
predict.py — Inferencia predictiva del PSBPM (version 3)
========================================================

Consume las trazas MCMC de UNA cadena y UNA componente FPCA, y entrega la
distribucion predictiva completa: momentos, muestras, densidad e intervalos.

Que cambia respecto de la version 2 y por que
---------------------------------------------
La version 2 devolvia, bajo `return_std=True`, la cantidad

    sd_post( E[y | x] ),

es decir la incertidumbre POSTERIOR SOBRE EL CENTRO de la predictiva. Esa
cantidad es correcta y util, pero NO es la desviacion predictiva: no incluye ni
la dispersion de la mezcla entre componentes ni el ruido intra-componente
tau_h^{-1}. Emplearla para construir bandas y medir cobertura produce
intervalos sistematicamente demasiado angostos, y la subcobertura resultante se
confunde con un fracaso del modelo cuando en realidad es un error de
construccion del intervalo.

La descomposicion correcta es la ley de varianza total sobre el posterior. Con
la mezcla en la iteracion t,

    m_t(x)   = sum_h pi_h(x) mu_h(x),
    s2_t(x)  = sum_h pi_h(x) ( mu_h(x)^2 + 1/tau_h ) - m_t(x)^2,

se obtiene

    E[y | x]   = E_t[ m_t(x) ],
    Var[y | x] = E_t[ s2_t(x) ]  +  Var_t[ m_t(x) ].
                 \_____________/    \______________/
                  dispersion de      incertidumbre
                  la predictiva      posterior sobre
                  dentro de cada     el centro
                  iteracion          (lo unico que
                                      media la v2)

`momentos_predictivos` entrega los tres terminos por separado, de modo que la
proporcion entre ellos es en si misma un diagnostico: si la varianza posterior
sobre el centro domina, la cadena no ha explorado suficiente o los datos son
escasos; si domina la dispersion de la mezcla, la incertidumbre es irreducible
del proceso.

Por que ademas hace falta MUESTREAR
-----------------------------------
Los dos primeros momentos no identifican una mezcla. El modelo propuesto existe
justamente para capturar multimodalidad y asimetria condicionales, de modo que
resumir su predictiva por (media, desviacion) descarta la evidencia que el
estudio busca producir: bajo la aproximacion gaussiana de dos momentos, los
escenarios con cambio de regimen o innovaciones asimetricas resultan
indistinguibles del escenario lineal gaussiano. Las reglas de puntuacion
propias del proyecto ---CRPS muestral, puntaje de energia, PIT--- requieren
extracciones de la predictiva, no sus momentos. `muestrear` las entrega.

Escala
------
El predictor es agnostico a la escala: opera en aquella en que se entrenaron
las trazas. Si el muestreador uso scores estandarizados, todas las salidas
estan en escala estandarizada; la vuelta a la escala de las curvas es
responsabilidad de `propagation.py`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

__all__ = ["PSBPPredictor", "pesos_probit", "medias_componente"]


# ==========================================================================
# NUCLEO: PESOS STICK-BREAKING PROBIT Y MEDIAS POR COMPONENTE
# ==========================================================================

def pesos_probit(Xnoint: np.ndarray, alphah: np.ndarray,
                 psijh: np.ndarray, Gammajh: np.ndarray) -> np.ndarray:
    """
    Pesos pi_h(x) del stick-breaking probit para una iteracion.

        v_h(x)  = Phi( alpha_h - sum_j psi_hj |x_j - Gamma_hj| )
        pi_h(x) = v_h(x) prod_{l<h} (1 - v_l(x)),   pi_N(x) = prod_l (1 - v_l(x))

    El producto acumulado se evalua en logaritmos: con N grande el producto
    directo subdesborda mucho antes que su logaritmo, y el peso de las ultimas
    componentes ---las que absorben la cola de la mezcla--- se anularia.

    Xnoint  : (n, p) covariables SIN intercepto
    alphah  : (N-1,)
    psijh   : (N-1, p)
    Gammajh : (N-1, p)
    Retorna : (n, N)
    """
    n = Xnoint.shape[0]
    N = alphah.shape[0] + 1

    diff = np.abs(Xnoint[None, :, :] - Gammajh[:, None, :])       # (N-1, n, p)
    eta = alphah[:, None] - np.sum(psijh[:, None, :] * diff, axis=2)
    vhx = norm.cdf(eta)                                            # (N-1, n)

    log_um = np.log(np.clip(1.0 - vhx, 1e-300, 1.0))
    cumlog = np.cumsum(log_um, axis=0)                             # (N-1, n)

    phx = np.zeros((N, n))
    phx[0] = vhx[0]
    if N > 1:
        phx[1:N - 1] = vhx[1:N - 1] * np.exp(cumlog[:N - 2])
        phx[N - 1] = np.exp(cumlog[N - 2])
    return phx.T                                                   # (n, N)


def medias_componente(X: np.ndarray, beta0h: np.ndarray,
                      betajh: np.ndarray) -> np.ndarray:
    """
    Media condicional por componente: mu_h(x) = beta_h0 + x' beta_h.

    X      : (n, p+1) con intercepto en la primera columna
    beta0h : (N,)
    betajh : (N, p)
    Retorna: (n, N)
    """
    return X[:, 0:1] * beta0h[None, :] + X[:, 1:] @ betajh.T


# ==========================================================================
# PREDICTOR
# ==========================================================================

class PSBPPredictor:
    """
    Distribucion predictiva de un PSBPM univariado a partir de sus trazas.

    Parametros
    ----------
    traces : dict
        Claves requeridas: betajhout (T, N, p), beta0hout (T, N),
        tauhout (T, N), alphahout (T, N-1), psijhout (T, N-1, p),
        Gammajhout (T, N-1, p). `osumout` es opcional.
    burn : int
        Iteraciones descartadas como calentamiento.

    Atributos
    ---------
    n_post_, n_components_, n_features_
    """

    def __init__(self, traces: Dict[str, np.ndarray], burn: int):
        self._validar(traces, burn)
        self.burn = int(burn)
        post = slice(self.burn, None)

        self._betajh = np.asarray(traces["betajhout"][post], dtype=float)
        self._beta0h = np.asarray(traces["beta0hout"][post], dtype=float)
        self._tauh = np.asarray(traces["tauhout"][post], dtype=float)
        self._alphah = np.asarray(traces["alphahout"][post], dtype=float)
        self._psijh = np.asarray(traces["psijhout"][post], dtype=float)
        self._Gammajh = np.asarray(traces["Gammajhout"][post], dtype=float)
        self._osum = (np.asarray(traces["osumout"][post], dtype=float)
                      if "osumout" in traces else None)

        self.n_post_ = self._betajh.shape[0]
        self.n_components_ = self._betajh.shape[1]
        self.n_features_ = self._betajh.shape[2]

        if np.any(self._tauh <= 0):
            raise ValueError(
                "tauhout contiene precisiones no positivas: la varianza "
                "intra-componente 1/tau_h no esta definida.")

    # ------------------------------------------------------------------
    @staticmethod
    def _validar(traces: Dict[str, np.ndarray], burn: int) -> None:
        req = {"betajhout", "beta0hout", "tauhout", "alphahout",
               "psijhout", "Gammajhout"}
        falta = req - set(traces.keys())
        if falta:
            raise KeyError(f"traces le faltan claves: {sorted(falta)}")
        nsim = traces["betajhout"].shape[0]
        if not (0 <= burn < nsim):
            raise ValueError(f"burn={burn} fuera de rango [0, {nsim}).")

    def _validar_X(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D; recibido {X.shape}.")
        if X.shape[1] != self.n_features_ + 1:
            raise ValueError(
                f"X tiene {X.shape[1]} columnas; se esperan "
                f"{self.n_features_ + 1} (intercepto + {self.n_features_}).")
        if not np.allclose(X[:, 0], 1.0):
            raise ValueError("Se espera X[:, 0] = 1 (columna de intercepto).")
        return X

    # ------------------------------------------------------------------
    # MOMENTOS POR ITERACION
    # ------------------------------------------------------------------
    def _momentos_por_iteracion(self, X: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Media y varianza de la mezcla en cada iteracion posterior.

        Retorna (m, s2), ambos (T, n):
            m[t, i]  = sum_h pi_h(x_i) mu_h(x_i)
            s2[t, i] = sum_h pi_h(x_i) (mu_h^2 + 1/tau_h) - m[t, i]^2
        """
        Xn = X[:, 1:]
        n, T = X.shape[0], self.n_post_
        m = np.empty((T, n))
        s2 = np.empty((T, n))

        for t in range(T):
            phx = pesos_probit(Xn, self._alphah[t], self._psijh[t],
                               self._Gammajh[t])                    # (n, N)
            mu = medias_componente(X, self._beta0h[t], self._betajh[t])
            var_h = 1.0 / self._tauh[t]                             # (N,)
            m_t = np.sum(phx * mu, axis=1)
            e2_t = np.sum(phx * (mu ** 2 + var_h[None, :]), axis=1)
            m[t] = m_t
            # El clip absorbe el error de redondeo cuando la mezcla degenera en
            # una sola componente y E[y^2] - E[y]^2 cae por debajo de cero.
            s2[t] = np.clip(e2_t - m_t ** 2, 0.0, None)
        return m, s2

    # ------------------------------------------------------------------
    # API PUBLICA
    # ------------------------------------------------------------------
    def momentos_predictivos(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Momentos de la predictiva con la descomposicion de varianza total.

        Retorna un diccionario con arreglos (n,):
            media        : E[y | x]
            sd           : desviacion PREDICTIVA (la que corresponde usar en
                           bandas y cobertura)
            var          : su cuadrado
            sd_intra     : sqrt( E_t[s2_t] ), dispersion de la mezcla
            sd_centro    : sqrt( Var_t[m_t] ), incertidumbre posterior sobre el
                           centro. Es la unica cantidad que devolvia la v2 bajo
                           el nombre `y_std`.
            frac_centro  : Var_t[m_t] / Var[y|x], en [0, 1]. Valores altos
                           indican que la incertidumbre esta dominada por el
                           desconocimiento de los parametros y no por la
                           variabilidad del proceso.
        """
        X = self._validar_X(X)
        m, s2 = self._momentos_por_iteracion(X)

        media = m.mean(axis=0)
        var_intra = s2.mean(axis=0)
        var_centro = m.var(axis=0, ddof=1) if self.n_post_ > 1 else np.zeros_like(media)
        var = var_intra + var_centro
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(var > 0, var_centro / var, np.nan)

        return {
            "media": media,
            "var": var,
            "sd": np.sqrt(var),
            "sd_intra": np.sqrt(var_intra),
            "sd_centro": np.sqrt(var_centro),
            "frac_centro": frac,
        }

    def predict(self, X: np.ndarray, return_std: bool = False
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prediccion puntual E[y|x].

        Con `return_std=True` retorna la desviacion PREDICTIVA completa, no la
        del centro. Este es el cambio de contrato respecto de la v2: quien
        necesite la incertidumbre posterior sobre el centro debe pedirla
        explicitamente como `sd_centro` en `momentos_predictivos`.
        """
        mom = self.momentos_predictivos(X)
        return (mom["media"], mom["sd"]) if return_std else mom["media"]

    # ------------------------------------------------------------------
    def muestrear(self, X: np.ndarray, extracciones_por_iteracion: int = 1,
                  seed: Optional[int] = None) -> np.ndarray:
        """
        Extrae muestras de la distribucion predictiva posterior.

        Para cada iteracion t se extrae la componente h ~ Categorica(pi(x)) y
        despues y ~ N(mu_h(x), tau_h^{-1}). Recorrer las iteraciones y extraer
        de la mezcla correspondiente a cada una produce muestras de la
        predictiva posterior, que es la mezcla de las predictivas condicionales
        a los parametros ponderada por el posterior.

        Retorna (S, n) con S = n_post * extracciones_por_iteracion. Esta forma
        es la que esperan `crps_muestral`, `pit_muestral` e
        `intervalo_muestral` del paquete `fit`.
        """
        X = self._validar_X(X)
        Xn = X[:, 1:]
        n, T = X.shape[0], self.n_post_
        d = int(extracciones_por_iteracion)
        if d < 1:
            raise ValueError("extracciones_por_iteracion debe ser >= 1.")

        rng = np.random.default_rng(seed)
        muestras = np.empty((T * d, n))

        for t in range(T):
            phx = pesos_probit(Xn, self._alphah[t], self._psijh[t],
                               self._Gammajh[t])                    # (n, N)
            mu = medias_componente(X, self._beta0h[t], self._betajh[t])
            sd_h = 1.0 / np.sqrt(self._tauh[t])                     # (N,)

            # Muestreo categorico vectorizado por inversion de la acumulada.
            acum = np.cumsum(phx, axis=1)
            acum = acum / acum[:, -1:]          # renormaliza el truncamiento en N
            u = rng.random((d, n, 1))
            h = (u > acum[None, :, :]).sum(axis=2)                  # (d, n)

            idx = np.arange(n)[None, :]
            centros = mu[idx, h]                                   # (d, n)
            escalas = sd_h[h]                                      # (d, n)
            muestras[t * d:(t + 1) * d] = centros + escalas * rng.standard_normal((d, n))

        return muestras

    # ------------------------------------------------------------------
    def intervalo(self, X: np.ndarray, nivel: float = 0.95,
                  metodo: str = "muestral",
                  extracciones_por_iteracion: int = 1,
                  seed: Optional[int] = None):
        """
        Intervalo predictivo central de nivel `nivel`.

        metodo='muestral'  : cuantiles empiricos de la predictiva. Es el
                             adecuado para una mezcla, porque respeta la
                             asimetria y la multimodalidad.
        metodo='gaussiano' : media +/- z * sd predictiva. Se ofrece solo como
                             referencia: impone simetria y unimodalidad, que es
                             exactamente lo que el modelo busca no suponer.

        Retorna (li, ls), ambos (n,).
        """
        if not (0.0 < nivel < 1.0):
            raise ValueError("nivel debe estar en (0, 1).")
        alpha = (1.0 - nivel) / 2.0

        if metodo == "muestral":
            Z = self.muestrear(X, extracciones_por_iteracion, seed)
            return (np.quantile(Z, alpha, axis=0),
                    np.quantile(Z, 1.0 - alpha, axis=0))
        if metodo == "gaussiano":
            mom = self.momentos_predictivos(X)
            z = float(norm.ppf(1.0 - alpha))
            return (mom["media"] - z * mom["sd"], mom["media"] + z * mom["sd"])
        raise ValueError(f"metodo='{metodo}' invalido; use 'muestral' o 'gaussiano'.")

    # ------------------------------------------------------------------
    def densidad(self, X: np.ndarray, y_grid: np.ndarray,
                 por_iteracion: bool = False) -> np.ndarray:
        """
        Densidad predictiva f(y|x) evaluada en `y_grid`, en forma cerrada.

        Se obtiene promediando sobre las iteraciones la densidad de la mezcla,
        sin recurrir a muestras, de modo que sirve tanto para graficar la
        predictiva como para el puntaje logaritmico exacto via
        `fit.lps_desde_log_densidad`.

        Retorna (n, len(y_grid)), o (T, n, len(y_grid)) si `por_iteracion`.
        """
        X = self._validar_X(X)
        Xn = X[:, 1:]
        y_grid = np.asarray(y_grid, dtype=float).ravel()
        n, T, Q = X.shape[0], self.n_post_, y_grid.size

        acumulada = np.zeros((n, Q))
        por_it = np.empty((T, n, Q)) if por_iteracion else None

        for t in range(T):
            phx = pesos_probit(Xn, self._alphah[t], self._psijh[t],
                               self._Gammajh[t])                    # (n, N)
            mu = medias_componente(X, self._beta0h[t], self._betajh[t])
            sd_h = 1.0 / np.sqrt(self._tauh[t])                     # (N,)
            z = (y_grid[None, None, :] - mu[:, :, None]) / sd_h[None, :, None]
            dens = np.sum(
                phx[:, :, None] * np.exp(-0.5 * z ** 2)
                / (sd_h[None, :, None] * np.sqrt(2.0 * np.pi)), axis=1)
            acumulada += dens
            if por_iteracion:
                por_it[t] = dens

        return por_it if por_iteracion else acumulada / T

    # ------------------------------------------------------------------
    def probabilidades_inclusion(self) -> Optional[np.ndarray]:
        """
        Probabilidad posterior de inclusion por covariable, si `osumout` esta
        disponible. `osumout` indica exclusion, de modo que la inclusion es su
        complemento.
        """
        if self._osum is None:
            return None
        return 1.0 - self._osum.mean(axis=0)
