"""
functions_standarize.py
========================
Clase de estandarización para coeficientes de series de tiempo funcionales. 

Contrato con el modelo
----------------------
    scaler  = FunctionalStandarizer(method="zscore")
    THETA_s = scaler.fit_transform(THETA)       # dentro de PSBPFunctional.fit()
    pred    = scaler.inverse_transform(pred_s)  # dentro de PSBPFunctional.predict()

Patrón: fit() / transform() / fit_transform() / inverse_transform()

Métodos
-------
  "zscore"  : (x - mean) / std   por columna   [default]
  "minmax"  : (x - min) / range  por columna
  "robust"  : (x - mediana) / IQR por columna
  "none"    : identidad
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

ScalerMethod = Literal["zscore", "minmax", "robust", "none"]


@dataclass
class FunctionalStandarizer:
    """
    Estandarización de coeficientes funcionales THETA (T, K).

    Parameters
    ----------
    method : str  —  método de estandarización (ver módulo docstring)

    Attributes (tras fit())
    -----------------------
    center_    : np.ndarray (K,)
    scale_     : np.ndarray (K,)
    K_         : int
    is_fitted_ : bool
    """
    method: ScalerMethod = "zscore"

    center_:    np.ndarray | None = field(default=None, repr=False)
    scale_:     np.ndarray | None = field(default=None, repr=False)
    K_:         int | None        = field(default=None, repr=False)
    is_fitted_: bool              = field(default=False, repr=False)

    # ── Interfaz pública ──────────────────────────────────────────────

    def fit(self, THETA: np.ndarray) -> "FunctionalStandarizer":
        """
        Calcula y almacena los parámetros de estandarización.

        Parameters
        ----------
        THETA : (T, K)  —  coeficientes en escala original
        """
        self._validate(THETA)
        T, K = THETA.shape
        self.K_ = K

        if self.method == "zscore":
            self.center_ = THETA.mean(axis=0)
            self.scale_  = THETA.std(axis=0, ddof=0)

        elif self.method == "minmax":
            self.center_ = THETA.min(axis=0)
            self.scale_  = THETA.max(axis=0) - self.center_

        elif self.method == "robust":
            self.center_ = np.median(THETA, axis=0)
            self.scale_  = np.percentile(THETA, 75, axis=0) \
                         - np.percentile(THETA, 25, axis=0)

        elif self.method == "none":
            self.center_ = np.zeros(K)
            self.scale_  = np.ones(K)

        else:
            raise ValueError(f"Método '{self.method}' no reconocido. "
                             "Opciones: zscore, minmax, robust, none.")

        # Columnas constantes → evitar división por cero
        self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        self.is_fitted_ = True
        return self

    def transform(self, THETA: np.ndarray) -> np.ndarray:
        """Aplica la estandarización con parámetros de fit(). Devuelve (T, K)."""
        self._check_fitted()
        self._validate(THETA, expected_K=self.K_)
        return (THETA - self.center_) / self.scale_

    def fit_transform(self, THETA: np.ndarray) -> np.ndarray:
        """fit() + transform() en un paso. Devuelve THETA_s (T, K)."""
        return self.fit(THETA).transform(THETA)

    def inverse_transform(self, THETA_s: np.ndarray) -> np.ndarray:
        """
        Invierte la estandarización → escala original.
        Usada en PSBPFunctional.predict() para devolver predicciones
        en la misma escala que los datos de entrada.
        """
        self._check_fitted()
        return THETA_s * self.scale_ + self.center_

    # ── Serialización ─────────────────────────────────────────────────

    def get_params(self) -> dict:
        """
        Devuelve parámetros como dict para serialización junto con el modelo.

        Example
        -------
        >>> saved = {"scaler": scaler.get_params(), "chains": model.chains_}
        >>> pickle.dump(saved, open("artefact/simulaciones/models/run01.pkl", "wb"))
        """
        self._check_fitted()
        return {
            "method":  self.method,
            "center_": self.center_.copy(),
            "scale_":  self.scale_.copy(),
            "K_":      self.K_,
        }

    @classmethod
    def from_params(cls, params: dict) -> "FunctionalStandarizer":
        """
        Reconstruye el scaler desde parámetros guardados.
        No requiere acceso a los datos originales.

        Example
        -------
        >>> scaler = FunctionalStandarizer.from_params(saved["scaler"])
        >>> pred = scaler.inverse_transform(pred_s)
        """
        obj = cls(method=params["method"])
        obj.center_    = np.asarray(params["center_"])
        obj.scale_     = np.asarray(params["scale_"])
        obj.K_         = int(params["K_"])
        obj.is_fitted_ = True
        return obj

    def summary(self) -> None:
        """Imprime un resumen de los parámetros ajustados."""
        self._check_fitted()
        print(f"FunctionalStandarizer | method='{self.method}' | K={self.K_}")
        print(f"  center (primeros 5): {np.round(self.center_[:5], 4)}")
        print(f"  scale  (primeros 5): {np.round(self.scale_[:5],  4)}")

    def __repr__(self) -> str:
        s = f"fitted(K={self.K_})" if self.is_fitted_ else "not fitted"
        return f"FunctionalStandarizer(method='{self.method}', {s})"

    # ── Validaciones ──────────────────────────────────────────────────

    @staticmethod
    def _validate(THETA: np.ndarray, expected_K: int | None = None) -> None:
        if not isinstance(THETA, np.ndarray):
            raise TypeError(f"Se esperaba np.ndarray, recibido: {type(THETA).__name__}.")
        if THETA.ndim != 2:
            raise ValueError(f"THETA debe ser 2D (T, K). Shape: {THETA.shape}.")
        if THETA.shape[0] < 2:
            raise ValueError("THETA necesita al menos 2 filas.")
        if expected_K is not None and THETA.shape[1] != expected_K:
            raise ValueError(
                f"K inconsistente: ajustado con K={expected_K}, recibido K={THETA.shape[1]}."
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Scaler no ajustado. Llamar a fit() o fit_transform() primero."
            )