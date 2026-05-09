"""
functions_repre_functional.py
==============================
Clase de representación funcional: curvas discretas → coeficientes THETA (T, K).

Contrato con el modelo
----------------------
    repre   = FunctionalRepresentation(method="bspline", n_basis=10)
    THETA   = repre.fit_transform(Y, grid)    # dentro de PSBPFunctional.fit()
    Y_hat   = repre.reconstruct(THETA)        # reconstrucción para diagnóstico

    # Si los datos ya son coeficientes, usar method="precomputed":
    repre   = FunctionalRepresentation(method="precomputed")
    THETA   = repre.fit_transform(THETA_raw)  # paso a través

Métodos soportados
------------------
  "bspline"     : B-splines cúbicos (skfda)
  "fourier"     : base de Fourier (skfda)
  "fpca"        : FPCA — scores como coeficientes (skfda)
  "precomputed" : los datos ya son coeficientes (T, K) — identidad

Nota sobre FPCA
---------------
  En FPCA, n_basis controla n_components (número de FPCs retenidas).
  El objeto FPCA ajustado se guarda en self.basis_ para permitir:
    - reconstruct(): proyección inversa
    - varianza explicada: self.fpca_.explained_variance_ratio_
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Any


RepreMethod = Literal["bspline", "fourier", "fpca", "precomputed"]


@dataclass
class FunctionalRepresentation:
    """
    Representación funcional de curvas discretas.

    Convierte Y (T, G) → THETA (T, K), donde:
      T = instantes de tiempo (observaciones)
      G = puntos de grilla de evaluación
      K = número de coeficientes (funciones de base o FPCs)

    Parameters
    ----------
    method  : método de representación (ver módulo docstring)
    n_basis : número de funciones de base (o FPCs para fpca)
    order   : orden del spline (default=4, cúbico). Solo para "bspline".
    domain  : (a, b) dominio de las curvas. None → inferido de grid en fit().
    center_fpca : bool. Si True, centra las curvas antes de FPCA. Default True.

    Attributes (tras fit())
    -----------------------
    grid_     : np.ndarray (G,)     grilla de evaluación usada en fit()
    basis_    : objeto base ajustado (BSplineBasis, FourierBasis, o FPCA)
    phi_      : np.ndarray (K, G)   funciones de base evaluadas en grid_
                                    (None para fpca — usar basis_ directamente)
    K_        : int                 número de coeficientes resultantes
    is_fitted_: bool
    """

    method:       RepreMethod = "bspline"
    n_basis:      int         = 10
    order:        int         = 4
    domain:       tuple[float, float] | None = None
    center_fpca:  bool        = True

    # Estado interno
    grid_:     np.ndarray | None = field(default=None, repr=False)
    basis_:    Any               = field(default=None, repr=False)
    phi_:      np.ndarray | None = field(default=None, repr=False)
    K_:        int | None        = field(default=None, repr=False)
    is_fitted_: bool             = field(default=False, repr=False)

    # ── Interfaz pública ──────────────────────────────────────────────

    def fit(self, Y: np.ndarray, grid: np.ndarray | None = None) -> "FunctionalRepresentation":
        """
        Ajusta la base funcional a los datos.

        Parameters
        ----------
        Y    : np.ndarray (T, G)  —  curvas evaluadas en grilla
               Para method="precomputed": Y es THETA (T, K) directamente.
        grid : np.ndarray (G,)    —  puntos de evaluación.
               Requerido para todos los métodos excepto "precomputed".

        Returns
        -------
        self
        """
        if self.method == "precomputed":
            self._fit_precomputed(Y)
        else:
            if grid is None:
                raise ValueError(
                    "grid es requerido para method != 'precomputed'."
                )
            grid = np.asarray(grid, dtype=float)
            self._validate_YG(Y, grid)
            self.grid_ = grid

            domain = self.domain or (float(grid.min()), float(grid.max()))

            if self.method == "bspline":
                self._fit_bspline(Y, grid, domain)
            elif self.method == "fourier":
                self._fit_fourier(Y, grid, domain)
            elif self.method == "fpca":
                self._fit_fpca(Y, grid, domain)
            else:
                raise ValueError(
                    f"Método '{self.method}' no reconocido. "
                    "Opciones: bspline, fourier, fpca, precomputed."
                )

        self.is_fitted_ = True
        return self

    def transform(self, Y: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
        """
        Proyecta curvas en la base ajustada → coeficientes THETA (T, K).

        Parameters
        ----------
        Y    : (T_new, G)  —  nuevas curvas en la misma grilla
        grid : solo necesario si method="fpca" (para crear FDataGrid)

        Returns
        -------
        THETA : np.ndarray (T_new, K)
        """
        self._check_fitted()

        if self.method == "precomputed":
            return np.asarray(Y, dtype=float)

        if self.method in ("bspline", "fourier"):
            # Proyección por mínimos cuadrados: THETA = Y @ phi_.T @ inv(phi_ @ phi_.T)
            # phi_ : (K, G)
            phi = self.phi_   # (K, G)
            A   = phi @ phi.T           # (K, K)
            B   = Y @ phi.T             # (T, K)
            THETA = np.linalg.solve(A, B.T).T  # (T, K)
            return THETA

        elif self.method == "fpca":
            grid_used = grid if grid is not None else self.grid_
            return self._transform_fpca(Y, grid_used)

    def fit_transform(
        self,
        Y: np.ndarray,
        grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Ajusta y transforma en un paso. Devuelve THETA (T, K).

        Uso en el modelo
        ----------------
        >>> repre  = FunctionalRepresentation(method="bspline", n_basis=10)
        >>> THETA  = repre.fit_transform(Y, grid)   # PSBPFunctional.fit()
        """
        return self.fit(Y, grid).transform(Y, grid)

    def reconstruct(self, THETA: np.ndarray) -> np.ndarray:
        """
        Reconstruye curvas desde coeficientes: THETA (T, K) → Y_hat (T, G).
        Usado para diagnóstico del error de aproximación funcional.

        Parameters
        ----------
        THETA : np.ndarray (T, K)

        Returns
        -------
        Y_hat : np.ndarray (T, G)
        """
        self._check_fitted()

        if self.method == "precomputed":
            raise RuntimeError(
                "reconstruct() no disponible para method='precomputed': "
                "no hay base funcional almacenada."
            )

        if self.method in ("bspline", "fourier"):
            # Y_hat = THETA @ phi_   →  (T, K) @ (K, G) = (T, G)
            return THETA @ self.phi_

        elif self.method == "fpca":
            return self._reconstruct_fpca(THETA)

    def reconstruction_error(
        self, Y: np.ndarray, grid: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Calcula el error de reconstrucción funcional en norma L2 discreta.

        Returns
        -------
        dict con claves: "rmse_mean", "rmse_std", "rel_error_mean"
        """
        self._check_fitted()
        THETA = self.transform(Y, grid)
        Y_hat = self.reconstruct(THETA)
        residuals = Y - Y_hat
        rmse_per_curve = np.sqrt((residuals ** 2).mean(axis=1))  # (T,)
        norm_per_curve = np.sqrt((Y ** 2).mean(axis=1))
        return {
            "rmse_mean":      float(rmse_per_curve.mean()),
            "rmse_std":       float(rmse_per_curve.std()),
            "rel_error_mean": float((rmse_per_curve / (norm_per_curve + 1e-12)).mean()),
        }

    # ── Método "bspline" ──────────────────────────────────────────────

    def _fit_bspline(
        self,
        Y: np.ndarray,
        grid: np.ndarray,
        domain: tuple[float, float],
    ) -> None:
        try:
            from skfda.representation.basis import BSplineBasis
        except ImportError:
            raise ImportError(
                "skfda es requerido para method='bspline'. "
                "Instalar con: pip install scikit-fda"
            )

        basis = BSplineBasis(
            domain_range=domain,
            n_basis=self.n_basis,
            order=self.order,
        )
        self.basis_ = basis

        # FIX: normalizar orientación de phi_ independientemente de la versión de skfda.
        # skfda puede devolver (K, G, 1) o (G, K, 1) según la versión.
        # Después de squeeze obtenemos (K, G) o (G, K); nos aseguramos de que
        # el primer eje sea siempre K antes de almacenar en phi_.
        phi_sq    = np.squeeze(basis(grid))          # (K, G) o (G, K)
        self.phi_ = phi_sq if phi_sq.shape[0] == self.n_basis else phi_sq.T  # → (K, G)
        self.K_   = self.n_basis

    # ── Método "fourier" ──────────────────────────────────────────────

    def _fit_fourier(
        self,
        Y: np.ndarray,
        grid: np.ndarray,
        domain: tuple[float, float],
    ) -> None:
        try:
            from skfda.representation.basis import FourierBasis
        except ImportError:
            raise ImportError(
                "skfda es requerido para method='fourier'. "
                "Instalar con: pip install scikit-fda"
            )

        period = domain[1] - domain[0]
        basis  = FourierBasis(
            domain_range=domain,
            n_basis=self.n_basis,
            period=period,
        )
        self.basis_ = basis

        # FIX: misma corrección de orientación que en _fit_bspline.
        phi_sq    = np.squeeze(basis(grid))          # (K, G) o (G, K)
        self.phi_ = phi_sq if phi_sq.shape[0] == self.n_basis else phi_sq.T  # → (K, G)
        self.K_   = self.n_basis

    # ── Método "fpca" ─────────────────────────────────────────────────

    def _fit_fpca(
        self,
        Y: np.ndarray,
        grid: np.ndarray,
        domain: tuple[float, float],
    ) -> None:
        try:
            from skfda import FDataGrid
            from skfda.preprocessing.dim_reduction import FPCA
        except ImportError:
            raise ImportError(
                "skfda es requerido para method='fpca'. "
                "Instalar con: pip install scikit-fda"
            )

        fd = FDataGrid(data_matrix=Y, grid_points=grid, domain_range=domain)
        fpca = FPCA(n_components=self.n_basis)
        fpca.fit(fd)

        self.basis_ = fpca
        self.phi_   = None    # en FPCA no usamos phi_ directamente
        self.K_     = self.n_basis

    def _transform_fpca(self, Y: np.ndarray, grid: np.ndarray) -> np.ndarray:
        from skfda import FDataGrid
        domain = self.domain or (float(grid.min()), float(grid.max()))
        fd     = FDataGrid(data_matrix=Y, grid_points=grid, domain_range=domain)
        return self.basis_.transform(fd)   # (T, K)

    def _reconstruct_fpca(self, THETA: np.ndarray) -> np.ndarray:
        """
        Reconstrucción FPCA: Y_hat = mean + THETA @ components

        Los componentes del objeto FPCA de skfda son objetos FDataGrid.
        Se evalúan en self.grid_ para obtener la matriz de reconstrucción.
        """
        fpca    = self.basis_
        grid    = self.grid_
        G       = len(grid)
        K       = self.K_

        # Evaluar los K componentes en la grilla → (K, G)
        comps = np.zeros((K, G))
        for k in range(K):
            comps[k] = fpca.components_(grid)[k].squeeze()

        # Media funcional (si FPCA centró)
        if self.center_fpca and hasattr(fpca, "mean_"):
            mean_curve = fpca.mean_(grid).squeeze()    # (G,)
        else:
            mean_curve = np.zeros(G)

        # Y_hat = THETA @ comps + mean_curve
        return THETA @ comps + mean_curve   # (T, G)

    # ── Método "precomputed" ──────────────────────────────────────────

    def _fit_precomputed(self, THETA: np.ndarray) -> None:
        if not isinstance(THETA, np.ndarray) or THETA.ndim != 2:
            raise ValueError(
                "Para method='precomputed', Y debe ser np.ndarray 2D (T, K)."
            )
        self.K_    = THETA.shape[1]
        self.grid_ = None
        self.phi_  = None

    # ── Información ───────────────────────────────────────────────────

    def fpca_explained_variance(self) -> np.ndarray:
        """
        Devuelve la varianza explicada por cada FPC.
        Solo disponible para method='fpca'.
        """
        if self.method != "fpca":
            raise RuntimeError(
                "fpca_explained_variance() solo disponible para method='fpca'."
            )
        self._check_fitted()
        return self.basis_.explained_variance_ratio_

    def get_params(self) -> dict:
        """
        Devuelve parámetros de configuración para serialización.
        Nota: el objeto basis_ (skfda) debe serializarse por separado con pickle.
        """
        self._check_fitted()
        return {
            "method":      self.method,
            "n_basis":     self.n_basis,
            "order":       self.order,
            "domain":      self.domain,
            "center_fpca": self.center_fpca,
            "K_":          self.K_,
            "phi_":        self.phi_.copy() if self.phi_ is not None else None,
        }

    def summary(self) -> None:
        self._check_fitted()
        print(f"FunctionalRepresentation | method='{self.method}' | "
              f"n_basis={self.n_basis} | K={self.K_}")
        if self.grid_ is not None:
            print(f"  grid: {len(self.grid_)} puntos en "
                  f"[{self.grid_.min():.3f}, {self.grid_.max():.3f}]")
        if self.method == "fpca":
            try:
                evr = self.fpca_explained_variance()
                print(f"  varianza explicada acumulada: "
                      f"{np.cumsum(evr)[-1]*100:.1f}%")
            except Exception:
                pass

    def __repr__(self) -> str:
        s = f"fitted(K={self.K_})" if self.is_fitted_ else "not fitted"
        return f"FunctionalRepresentation(method='{self.method}', n_basis={self.n_basis}, {s})"

    # ── Validaciones ──────────────────────────────────────────────────

    @staticmethod
    def _validate_YG(Y: np.ndarray, grid: np.ndarray) -> None:
        if not isinstance(Y, np.ndarray) or Y.ndim != 2:
            raise ValueError(f"Y debe ser np.ndarray 2D (T, G). Shape: {Y.shape}.")
        if grid.ndim != 1:
            raise ValueError(f"grid debe ser 1D (G,). Shape: {grid.shape}.")
        if Y.shape[1] != len(grid):
            raise ValueError(
                f"Y tiene {Y.shape[1]} columnas pero grid tiene {len(grid)} puntos."
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "El objeto no está ajustado. "
                "Llamar a fit() o fit_transform() primero."
            )
