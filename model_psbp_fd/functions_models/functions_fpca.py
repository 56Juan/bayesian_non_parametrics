"""
functions_fpca.py
=================
Analisis de Componentes Principales Funcional en la metrica de L^2.

Cuando la representacion funcional se construye sobre un sistema de base no
ortonormal ---como las B-splines, cuya matriz de Gram W no es diagonal--- la
descomposicion espectral del operador de covarianza no se reduce a un problema
de autovalores ordinario sobre los coeficientes, sino a uno generalizado

    C u = lambda W u,

donde C es la covarianza de los coeficientes y W la matriz de Gram. La
presencia de W es la correccion por la no ortonormalidad del sistema: garantiza
que las autofunciones estimadas sean ortonormales en la metrica de L^2 y no en
la metrica euclidea de los coeficientes.

La implementacion resuelve el problema en su forma simetrizada

    (W^{1/2} C W^{1/2}) z = lambda z,     b = W^{-1/2} z,

numericamente estable por operar sobre una matriz simetrica.

Patron fit / transform
----------------------
La clase separa el ajuste de la aplicacion. Bajo el esquema de retencion
temporal, `fit` recibe unicamente el bloque de entrenamiento y `transform`
proyecta cualquier bloque sobre la base ya estimada. Esta separacion es la que
hace que la ausencia de fuga de informacion sea una propiedad de la clase y no
una disciplina que el usuario deba recordar en cada notebook.

Inversas de la representacion
-----------------------------
Por la ortonormalidad en metrica L^2 (B^T W B = I), la inversa del truncamiento
FPCA es B^T y no la pseudoinversa de W B: la segunda es la inversa por la
izquierda de norma euclidea minima, que no coincide con la proyeccion FPCA
cuando W != I. Con la inversa correcta, las dos rutas de reconstruccion

    X = mu + Xi Psi^T          (via autofunciones)
    Theta = mu_theta + Xi B^T,  X = Theta Phi^T     (via coeficientes de base)

producen exactamente la misma curva, propiedad que `verificar` comprueba.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..utils.quadrature import pesos_trapezoidales, gram

__all__ = ["FPCA_L2", "base_en_grilla"]


def base_en_grilla(fr, K: Optional[int] = None) -> np.ndarray:
    """
    Evalua el sistema de base de una `FunctionalRepresentation` en su grilla.

    La k-esima funcion base se obtiene por diferencia,

        phi_k = reconstruct(e_k) - reconstruct(0),

    y no como `reconstruct(e_k)` directamente. La razon no es cosmetica: si la
    representacion se ajusto con `center=True`, `reconstruct` es el mapa AFIN
    Theta Phi^T + media y no el lineal Theta Phi^T, de modo que aplicarlo a la
    identidad devuelve cada funcion base CONTAMINADA con la media funcional.
    La contaminacion se propaga entonces a la matriz de Gram, a las
    autofunciones y a los scores sin producir error alguno. Restar
    `reconstruct(0)` cancela el termino constante y devuelve la base correcta
    tanto si la representacion centra como si no.

    Retorna Phi de forma (G, K).
    """
    if K is None:
        K = int(getattr(fr, "K_", None) or getattr(fr, "n_basis"))
    base = np.asarray(fr.reconstruct(np.eye(K)), dtype=float)
    origen = np.asarray(fr.reconstruct(np.zeros((1, K))), dtype=float)
    return (base - origen).T


@dataclass
class FPCA_L2:
    """
    FPCA en metrica L^2 sobre coeficientes de una base no ortonormal.

    Parametros
    ----------
    n_components : int, opcional
        Numero de componentes a retener. Si es None debe fijarse despues del
        ajuste mediante `seleccionar_M` o `set_M`.

    Atributos tras `fit`
    --------------------
    W          : (K, K) matriz de Gram de la base en metrica L^2.
    cond_W     : float  numero de condicion de W. Se registra porque la raiz
                 inversa W^{-1/2} amplifica el ruido de las direcciones
                 asociadas a los autovalores menores de W; con bases muy
                 solapadas o K grande esa amplificacion degrada las
                 componentes de menor varianza, que son justamente las que se
                 truncan. Tener el valor permite auditar el efecto al comparar
                 escenarios con distinto K en lugar de descubrirlo tarde.
    mu_theta   : (K,)   media de los coeficientes del bloque de ajuste.
    B_full     : (K, K) coeficientes de todas las autofunciones.
    evals      : (K,)   autovalores en orden decreciente.
    var_ratio  : (K,)   proporcion de varianza por componente.
    var_cum    : (K,)   proporcion acumulada.
    """

    n_components: Optional[int] = None

    # Estado del ajuste
    Phi: np.ndarray = field(default=None, repr=False)
    tau: np.ndarray = field(default=None, repr=False)
    W: np.ndarray = field(default=None, repr=False)
    cond_W: Optional[float] = field(default=None, repr=False)
    mu_theta: np.ndarray = field(default=None, repr=False)
    B_full: np.ndarray = field(default=None, repr=False)
    evals: np.ndarray = field(default=None, repr=False)
    var_ratio: np.ndarray = field(default=None, repr=False)
    var_cum: np.ndarray = field(default=None, repr=False)
    n_ajuste: int = 0
    is_fitted_: bool = False

    # ------------------------------------------------------------------
    # AJUSTE
    # ------------------------------------------------------------------
    def fit(self, THETA_train: np.ndarray, Phi: np.ndarray,
            tau: np.ndarray) -> "FPCA_L2":
        """
        Estima la base FPCA a partir del bloque de ajuste.

        THETA_train : (n, K) coeficientes de las curvas de entrenamiento.
        Phi         : (G, K) base evaluada en la grilla.
        tau         : (G,)   grilla del dominio.

        Bajo retencion temporal, `THETA_train` debe contener EXCLUSIVAMENTE el
        bloque de entrenamiento: la media y la covarianza que aqui se estiman
        son los objetos que no deben ver el bloque de prueba.
        """
        THETA_train = np.atleast_2d(np.asarray(THETA_train, dtype=float))
        Phi = np.asarray(Phi, dtype=float)
        tau = np.asarray(tau, dtype=float)

        n, K = THETA_train.shape
        if n < 2:
            raise ValueError(f"Se requieren al menos 2 curvas para ajustar; n={n}.")
        if Phi.shape[1] != K:
            raise ValueError(
                f"Phi tiene {Phi.shape[1]} columnas y THETA {K}: la base y los "
                "coeficientes no corresponden al mismo sistema."
            )

        self.Phi, self.tau = Phi, tau
        self.W = gram(Phi, tau)

        self.mu_theta = THETA_train.mean(axis=0)
        Tc = THETA_train - self.mu_theta
        S = (Tc.T @ Tc) / (n - 1)

        # Raices de la matriz de Gram por descomposicion espectral.
        # El recorte de autovalores es RELATIVO al mayor de ellos y no
        # absoluto: un umbral fijo como 1e-12 es severo si W esta escalada en
        # 1e-6 e inocuo si lo esta en 1e+6, de modo que su efecto dependeria
        # de las unidades del dominio y no del condicionamiento real.
        evW, VW = np.linalg.eigh(self.W)
        self.cond_W = float(evW.max() / evW.min()) if evW.min() > 0 else np.inf
        piso = float(np.finfo(float).eps * max(evW.max(), 0.0) * self.W.shape[0])
        evW = np.clip(evW, max(piso, 1e-300), None)
        W_half = VW @ np.diag(np.sqrt(evW)) @ VW.T
        W_half_inv = VW @ np.diag(1.0 / np.sqrt(evW)) @ VW.T

        M_sym = W_half @ S @ W_half
        M_sym = 0.5 * (M_sym + M_sym.T)
        evals, U = np.linalg.eigh(M_sym)

        orden = np.argsort(evals)[::-1]
        self.evals = np.clip(evals[orden], 0.0, None)
        self.B_full = W_half_inv @ U[:, orden]

        total = self.evals.sum()
        self.var_ratio = self.evals / total if total > 0 else np.zeros_like(self.evals)
        self.var_cum = np.cumsum(self.var_ratio)

        self.n_ajuste = int(n)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # SELECCION DEL NUMERO DE COMPONENTES
    # ------------------------------------------------------------------
    def seleccionar_M(self, umbral: float = 0.99) -> int:
        """
        Menor numero de componentes cuya varianza acumulada alcanza `umbral`.

        Es una SUGERENCIA: el criterio de varianza explicada es monotono y no
        incorpora la dinamica temporal de los scores, de modo que la eleccion
        final corresponde al analista. `set_M` es el punto donde esa decision
        se registra.
        """
        self._check_fitted()
        if not (0.0 < umbral <= 1.0):
            raise ValueError("umbral debe estar en (0, 1].")
        return int(np.clip(np.searchsorted(self.var_cum, umbral) + 1,
                           1, self.evals.size))

    def set_M(self, M: int) -> "FPCA_L2":
        """Fija el numero de componentes retenidas."""
        self._check_fitted()
        K = self.evals.size
        if not (isinstance(M, (int, np.integer)) and 1 <= M <= K):
            raise ValueError(f"M debe ser entero en [1, {K}]; recibido {M!r}.")
        self.n_components = int(M)
        return self

    # ------------------------------------------------------------------
    # PROPIEDADES DERIVADAS
    # ------------------------------------------------------------------
    @property
    def M(self) -> int:
        self._check_M()
        return int(self.n_components)

    @property
    def B(self) -> np.ndarray:
        """(K, M) coeficientes de las autofunciones retenidas."""
        return self.B_full[:, :self.M]

    @property
    def Psi_grid(self) -> np.ndarray:
        """(G, M) autofunciones evaluadas en la grilla; ortonormales en L^2."""
        return self.Phi @ self.B

    @property
    def mu_grid(self) -> np.ndarray:
        """(G,) funcion media evaluada en la grilla."""
        return self.Phi @ self.mu_theta

    @property
    def lambdas(self) -> np.ndarray:
        """(M,) autovalores retenidos; varianza de cada score."""
        return self.evals[:self.M]

    # ------------------------------------------------------------------
    # TRANSFORMACIONES
    # ------------------------------------------------------------------
    def transform(self, THETA: np.ndarray) -> np.ndarray:
        """
        Proyecta coeficientes sobre la base ajustada -> scores (T, M).

        Admite bloques de entrenamiento o de prueba indistintamente: el
        centrado emplea siempre la media del bloque de ajuste.
        """
        self._check_M()
        THETA = np.atleast_2d(np.asarray(THETA, dtype=float))
        return (THETA - self.mu_theta) @ (self.W @ self.B)

    def inverse_transform(self, SCORES: np.ndarray) -> np.ndarray:
        """
        Scores -> coeficientes de la base (T, K), via Theta = mu_theta + Xi B^T.

        Emplea B^T y no la pseudoinversa de W B, que no coincide con la
        proyeccion FPCA cuando la base no es ortonormal.
        """
        self._check_M()
        SCORES = np.atleast_2d(np.asarray(SCORES, dtype=float))
        return SCORES @ self.B.T + self.mu_theta

    def reconstruct(self, SCORES: np.ndarray) -> np.ndarray:
        """Scores -> curvas en la grilla (T, G), via X = mu + Xi Psi^T."""
        self._check_M()
        SCORES = np.atleast_2d(np.asarray(SCORES, dtype=float))
        return self.mu_grid[None, :] + SCORES @ self.Psi_grid.T

    def fit_transform(self, THETA_train, Phi, tau, M=None) -> np.ndarray:
        """Ajusta con el bloque de entrenamiento y proyecta ese mismo bloque."""
        self.fit(THETA_train, Phi, tau)
        if M is not None:
            self.set_M(M)
        return self.transform(THETA_train)

    # ------------------------------------------------------------------
    # VERIFICACION
    # ------------------------------------------------------------------
    def verificar(self, THETA_train: Optional[np.ndarray] = None,
                  fr=None, tol: float = 1e-10) -> dict:
        """
        Comprueba las identidades que sostienen la construccion.

        - Ortonormalidad en L^2 de las autofunciones retenidas: Psi^T W_L2 Psi = I.
        - Ortonormalidad en la metrica de Gram: B^T W B = I.
        - Equivalencia de las dos rutas de reconstruccion.
        - Si se entrega `THETA_train`: media nula de los scores, varianza igual
          a los autovalores y ausencia de correlacion contemporanea. Estas tres
          propiedades son consecuencia del ajuste y solo deben verificarse en el
          bloque de entrenamiento; su desviacion en el bloque de prueba es
          diagnostica de deriva del proceso, no un error.
        - Si se entrega `fr`: linealidad de `reconstruct` como mapa Theta Phi^T.

        Puede invocarse antes de fijar el numero de componentes: en ese caso la
        verificacion emplea la base completa (M = K), lo que corresponde a un
        diagnostico del ajuste con independencia del truncamiento posterior.
        """
        self._check_fitted()
        M_previo = self.n_components
        if self.n_components is None:
            self.n_components = int(self.evals.size)
        try:
            return self._verificar_impl(THETA_train, fr, tol)
        finally:
            self.n_components = M_previo

    def _verificar_impl(self, THETA_train, fr, tol) -> dict:
        w = pesos_trapezoidales(self.tau)
        Psi = self.Psi_grid
        M = self.M
        K = int(self.evals.size)

        d = {
            "M": M,
            "K": K,
            "n_ajuste": self.n_ajuste,
            "cond_W": self.cond_W,
            "err_ortonormalidad_L2": float(
                np.abs(Psi.T @ (w[:, None] * Psi) - np.eye(M)).max()),
            "err_ortonormalidad_gram": float(
                np.abs(self.B.T @ self.W @ self.B - np.eye(M)).max()),
            "var_explicada": float(self.var_cum[M - 1]),
        }

        S_chk = np.eye(M)
        d["err_rutas_reconstruccion"] = float(np.abs(
            self.reconstruct(S_chk) - self.inverse_transform(S_chk) @ self.Phi.T
        ).max())

        if fr is not None:
            # Se emplea un THETA generico y no la identidad. Con Theta = I el
            # chequeo era una TAUTOLOGIA: `base_en_grilla` define Phi como la
            # imagen de la identidad bajo `reconstruct`, de modo que
            # `fr.reconstruct(I)` e `I @ Phi^T` coinciden por construccion aun
            # cuando `reconstruct` sea afin y no lineal. Con un Theta arbitrario
            # el termino constante deja de cancelarse y la violacion se detecta.
            rng = np.random.default_rng(0)
            Th = rng.standard_normal((max(3, min(8, K)), K))
            escala = float(np.abs(self.Phi).max()) or 1.0
            err_lin = float(np.abs(
                np.asarray(fr.reconstruct(Th), dtype=float) - Th @ self.Phi.T
            ).max())
            d["err_linealidad_reconstruct"] = err_lin
            d["err_linealidad_reconstruct_rel"] = err_lin / escala

        if THETA_train is not None:
            Sc = self.transform(THETA_train)
            sd_scores = np.sqrt(np.clip(self.lambdas, 1e-300, None))
            # Las dos cantidades siguientes tienen las unidades de los scores y
            # de su varianza respectivamente, de modo que no pueden compararse
            # contra la misma tolerancia que los errores adimensionales de
            # ortonormalidad: se reportan tambien en forma relativa, que es la
            # que decide `todo_ok`.
            err_media = float(np.abs(Sc.mean(axis=0)).max())
            err_var = float(np.abs(Sc.var(axis=0, ddof=1) - self.lambdas).max())
            d["err_media_scores"] = err_media
            d["err_media_scores_rel"] = float(
                np.max(np.abs(Sc.mean(axis=0)) / sd_scores))
            d["err_var_scores_vs_lambda"] = err_var
            d["err_var_scores_vs_lambda_rel"] = float(np.max(
                np.abs(Sc.var(axis=0, ddof=1) - self.lambdas)
                / np.clip(self.lambdas, 1e-300, None)))
            if M > 1:
                R = np.corrcoef(Sc, rowvar=False)
                d["err_correlacion_lag0"] = float(np.abs(R - np.eye(M)).max())
            else:
                d["err_correlacion_lag0"] = 0.0

        # `todo_ok` se evalua UNICAMENTE sobre cantidades adimensionales o
        # relativas. Incluir errores en las unidades de los datos hacia que el
        # veredicto dependiera de la amplitud de las curvas: con scores de
        # varianza grande un error relativo despreciable podia superar la
        # tolerancia, y con curvas de amplitud pequena un error relativo
        # inaceptable podia pasar inadvertido.
        adimensionales = (
            "err_ortonormalidad_L2",
            "err_ortonormalidad_gram",
            "err_rutas_reconstruccion",
            "err_correlacion_lag0",
            "err_linealidad_reconstruct_rel",
            "err_media_scores_rel",
            "err_var_scores_vs_lambda_rel",
        )
        evaluados = [d[k] for k in adimensionales if k in d]
        d["criterios_evaluados"] = list(
            k for k in adimensionales if k in d)
        d["todo_ok"] = bool(max(evaluados) < tol)
        return d

    def resumen_componentes(self):
        """DataFrame con autovalor, proporcion y acumulado por componente."""
        self._check_fitted()
        import pandas as pd
        return pd.DataFrame({
            "componente": np.arange(1, self.evals.size + 1),
            "autovalor": self.evals,
            "var_ratio": self.var_ratio,
            "var_acum": self.var_cum,
        })

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("FPCA_L2 no ajustado: llame a fit() primero.")

    def _check_M(self) -> None:
        self._check_fitted()
        if self.n_components is None:
            raise RuntimeError(
                "n_components no definido: use set_M(M) o seleccionar_M() "
                "antes de transformar o reconstruir."
            )
