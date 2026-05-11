"""
functions_repre_functional.py
=============================

Representación funcional de curvas para el pipeline PSBP-FD.

Pipeline conceptual:
    curvas Y (T, G) ──► FunctionalRepresentation ──► coeficientes THETA (T, K)
    (T = observaciones, G = puntos de grilla, K = coeficientes finales)

Métodos soportados:
    - "bspline"          : proyección sobre base B-spline.
    - "fourier"          : proyección sobre base de Fourier.
    - "fpca"             : Functional PCA directo sobre las curvas.
    - "bspline+fpca"     : suavizado B-spline → FPCA sobre coeficientes.
    - "fourier+fpca"     : suavizado Fourier → FPCA sobre coeficientes.

Proyección sobre bases (bspline/fourier):
    - "l2"       : proyección L² con integración (Simpson o trapezoidal).
                   THETA = G⁻¹ B,  G_jk = ∫φ_j φ_k dt,  B_j = ∫y φ_j dt.
    - "discrete" : proyección discreta (ignora espaciamiento de la grilla).
                   THETA = (ΦΦ')⁻¹ Φ Y'.

Reconstruction error en dos normas (siempre devuelve ambas):
    - L² integrada : ‖y - ŷ‖²_L² = ∫ (y(t) - ŷ(t))² dt.
    - Punto a punto: ‖y - ŷ‖²_disc = (1/G) Σ_g (y_g - ŷ_g)².

Convención de "frozen": instancias creadas vía `from_coefficients` no
tienen base subyacente; solo exponen los coeficientes y la información
mínima. Operaciones de reconstrucción y plot de bases no están disponibles.

Author: model_psbp_fd
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any, Optional, Dict, Tuple
import warnings

import numpy as np
from scipy.integrate import simpson


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

RepreMethod      = Literal["bspline", "fourier", "fpca",
                           "bspline+fpca", "fourier+fpca"]
ProjectionMethod = Literal["l2", "discrete"]
QuadratureMethod = Literal["trapezoid", "simpson"]


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionalRepresentation:
    """
    Representación funcional de curvas discretas para PSBP-FD.

    Parámetros
    ----------
    method : RepreMethod
        Método de representación. Ver módulo docstring para detalles.

    n_basis : int, default=20
        Número de funciones de base de la primera etapa.
            - Para 'bspline' y 'fourier': K final.
            - Para 'fpca' puro: ignorado.
            - Para 'bspline+fpca' / 'fourier+fpca': K intermedio (suavizado).

    n_basis_fpca : int, default=3
        Número de componentes FPCA finales.
            - Solo aplica si `method` contiene 'fpca'.
            - Para 'fpca' puro: K final.
            - Para encadenados: K final tras la reducción FPCA.

    order : int, default=4
        Orden del spline (4 = cúbico). Solo aplica a 'bspline'.

    domain : tuple[float, float], opcional
        Dominio de las curvas. Si None, se infiere de `grid` en `fit`.

    projection : {'l2', 'discrete'}, default='l2'
        Tipo de proyección para 'bspline' y 'fourier'.

    quadrature : {'trapezoid', 'simpson'}, default='trapezoid'
        Regla de cuadratura para integrales L². Simpson requiere G impar.

    center : bool, default=False
        Si True, centra las curvas (resta la media funcional) antes de
        proyectar/aplicar FPCA. FPCA siempre centra internamente; este
        flag solo afecta la representación de salida.

    Atributos post-fit
    ------------------
    grid_      : (G,)         grilla de evaluación
    domain_    : (a, b)       dominio efectivo
    K_         : int          dimensión final de los coeficientes
    mean_      : (G,)         media funcional de las curvas de fit
    phi_       : (K_first, G) base inicial evaluada en grid (None si FPCA puro)
    gram_      : (K_first, K_first)  matriz de Gram L² (solo si projection='l2')
    basis_     : objeto skfda      base de la primera etapa
    fpca_      : objeto FPCA       (None si no hay FPCA)
    is_fitted_ : bool
    """

    # ── Configuración (input del usuario) ────────────────────────────────────
    method:        RepreMethod        = "bspline"
    n_basis:       int                = 20
    n_basis_fpca:  int                = 3
    order:         int                = 4
    domain:        Optional[Tuple[float, float]] = None
    projection:    ProjectionMethod   = "l2"
    quadrature:    QuadratureMethod   = "trapezoid"
    center:        bool               = False

    # ── Estado interno (post-fit) ────────────────────────────────────────────
    grid_:      Optional[np.ndarray] = field(default=None, repr=False)
    domain_:    Optional[Tuple[float, float]] = field(default=None, repr=False)
    K_:         Optional[int]        = field(default=None, repr=False)
    mean_:      Optional[np.ndarray] = field(default=None, repr=False)
    phi_:       Optional[np.ndarray] = field(default=None, repr=False)
    gram_:      Optional[np.ndarray] = field(default=None, repr=False)
    basis_:     Any                  = field(default=None, repr=False)
    fpca_:      Any                  = field(default=None, repr=False)
    is_fitted_: bool                 = field(default=False, repr=False)
    is_frozen_: bool                 = field(default=False, repr=False)

    # ─────────────────────────────────────────────────────────────────────
    # Validación de configuración (al construir)
    # ─────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        valid_methods = {"bspline", "fourier", "fpca",
                         "bspline+fpca", "fourier+fpca"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method='{self.method}' inválido. Opciones: {sorted(valid_methods)}"
            )
        if self.projection not in {"l2", "discrete"}:
            raise ValueError(
                f"projection='{self.projection}' inválido. Use 'l2' o 'discrete'."
            )
        if self.quadrature not in {"trapezoid", "simpson"}:
            raise ValueError(
                f"quadrature='{self.quadrature}' inválido. "
                "Use 'trapezoid' o 'simpson'."
            )
        if self.n_basis < 1:
            raise ValueError(f"n_basis debe ser ≥ 1; recibido {self.n_basis}")
        if "fpca" in self.method and self.n_basis_fpca < 1:
            raise ValueError(
                f"n_basis_fpca debe ser ≥ 1; recibido {self.n_basis_fpca}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # API pública: fit / transform / fit_transform
    # ─────────────────────────────────────────────────────────────────────
    def fit(self, Y: np.ndarray, grid: np.ndarray) -> "FunctionalRepresentation":
        """
        Ajusta la representación funcional a `Y` evaluado en `grid`.

        Parámetros
        ----------
        Y : ndarray, shape (T, G)
            Curvas discretas. T = observaciones, G = puntos de grilla.
        grid : ndarray, shape (G,)
            Puntos de evaluación.

        Retorna
        -------
        self
        """
        Y = np.asarray(Y, dtype=float)
        grid = np.asarray(grid, dtype=float)
        self._validate_Y_grid(Y, grid)

        self.grid_   = grid
        self.domain_ = self.domain or (float(grid.min()), float(grid.max()))
        self.mean_   = Y.mean(axis=0)

        # Curvas a usar (centradas o no) para la etapa de base
        Y_work = (Y - self.mean_) if self.center else Y

        # Etapa 1: base (B-spline, Fourier o nada si FPCA puro)
        if self.method in ("bspline", "bspline+fpca"):
            self._fit_bspline(grid)
        elif self.method in ("fourier", "fourier+fpca"):
            self._fit_fourier(grid)
        elif self.method == "fpca":
            self._fit_fpca_direct(Y, grid)
        else:
            raise ValueError(f"method='{self.method}' no implementado.")

        # Etapa 2: FPCA encadenado, si aplica
        if self.method in ("bspline+fpca", "fourier+fpca"):
            theta_smooth = self._project_to_base(Y_work)   # (T, K_base)
            self._fit_fpca_on_coefs(theta_smooth)

        # Dimensión final
        if "fpca" in self.method:
            self.K_ = self.n_basis_fpca
        else:
            self.K_ = self.phi_.shape[0]

        self.is_fitted_ = True
        return self

    def transform(self, Y: np.ndarray,
                  grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Proyecta nuevas curvas en la base ajustada → THETA (T, K).

        Parámetros
        ----------
        Y    : (T_new, G)
        grid : (G,), opcional. Si None, usa la grilla del fit.

        Retorna
        -------
        THETA : (T_new, K_)
        """
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError(
                "Instancia frozen (creada vía from_coefficients). "
                "transform() no aplica; los coeficientes ya son la representación."
            )

        Y = np.asarray(Y, dtype=float)
        grid_used = self.grid_ if grid is None else np.asarray(grid, dtype=float)
        if grid is not None and not np.allclose(grid_used, self.grid_):
            warnings.warn(
                "grid distinto al usado en fit(); los coeficientes pueden no ser "
                "comparables a los del entrenamiento.",
                UserWarning,
            )

        if Y.ndim != 2 or Y.shape[1] != len(grid_used):
            raise ValueError(
                f"Y debe ser (T, G={len(grid_used)}); recibido {Y.shape}"
            )

        Y_work = (Y - self.mean_) if self.center else Y

        if self.method == "fpca":
            return self._transform_fpca_direct(Y_work, grid_used)

        theta_base = self._project_to_base(Y_work)        # (T, K_base)

        if self.method in ("bspline+fpca", "fourier+fpca"):
            # Proyecta los coeficientes de la base sobre los autovectores FPCA
            # Centrado interno de FPCA-on-coefs ya manejado en _fit_fpca_on_coefs.
            theta_centered = theta_base - self._fpca_mean_coefs
            return theta_centered @ self._fpca_components_coefs.T  # (T, K_fpca)

        return theta_base

    def fit_transform(self, Y: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Atajo: fit(Y, grid).transform(Y, grid)."""
        return self.fit(Y, grid).transform(Y, grid)

    # ─────────────────────────────────────────────────────────────────────
    # Reconstrucción y error
    # ─────────────────────────────────────────────────────────────────────
    def reconstruct(self, THETA: np.ndarray) -> np.ndarray:
        """
        Reconstruye curvas en escala original a partir de los coeficientes.

        Si `center=True` fue usado en fit, la media funcional se vuelve a
        sumar automáticamente.

        Parámetros
        ----------
        THETA : (T, K_)

        Retorna
        -------
        Y_hat : (T, G)
        """
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("reconstruct() no disponible para instancias frozen.")

        THETA = np.asarray(THETA, dtype=float)

        if self.method == "fpca":
            Y_hat = self._reconstruct_fpca_direct(THETA)
        elif self.method in ("bspline+fpca", "fourier+fpca"):
            # THETA (T, K_fpca) → coeficientes en base intermedia → curvas
            theta_base = THETA @ self._fpca_components_coefs + self._fpca_mean_coefs
            Y_hat = theta_base @ self.phi_                # (T, G)
        else:
            # B-spline o Fourier puro
            Y_hat = THETA @ self.phi_                     # (T, G)

        if self.center:
            Y_hat = Y_hat + self.mean_
        return Y_hat

    def reconstruction_error(self,
                             Y: np.ndarray,
                             grid: Optional[np.ndarray] = None
                             ) -> Dict[str, float]:
        """
        Error de reconstrucción en dos normas:

            - 'l2'       : ‖y - ŷ‖_L² = √(∫ (y(t) - ŷ(t))² dt)
            - 'pointwise': ‖y - ŷ‖_2 / √G  (RMSE punto a punto)

        Retorna
        -------
        dict con:
            'rmse_l2_mean',     'rmse_l2_std'
            'rmse_pw_mean',     'rmse_pw_std'
            'rel_l2_mean',      'rel_l2_std'
        """
        self._check_fitted()
        Y = np.asarray(Y, dtype=float)
        grid_used = self.grid_ if grid is None else np.asarray(grid, dtype=float)

        THETA = self.transform(Y, grid_used)
        Y_hat = self.reconstruct(THETA)
        residuals = Y - Y_hat                              # (T, G)

        # L² integrada
        l2_sq = self._integrate_L2(residuals ** 2, grid_used)  # (T,)
        l2_norm_y = np.sqrt(self._integrate_L2(Y ** 2, grid_used))
        rmse_l2 = np.sqrt(l2_sq)
        rel_l2  = rmse_l2 / (l2_norm_y + 1e-12)

        # Punto a punto
        rmse_pw = np.sqrt((residuals ** 2).mean(axis=1))

        return {
            "rmse_l2_mean":  float(rmse_l2.mean()),
            "rmse_l2_std":   float(rmse_l2.std()),
            "rmse_pw_mean":  float(rmse_pw.mean()),
            "rmse_pw_std":   float(rmse_pw.std()),
            "rel_l2_mean":   float(rel_l2.mean()),
            "rel_l2_std":    float(rel_l2.std()),
        }

    # ─────────────────────────────────────────────────────────────────────
    # FPCA: información de varianza y componentes
    # ─────────────────────────────────────────────────────────────────────
    def fpca_explained_variance(self) -> np.ndarray:
        """
        Varianza explicada por cada FPC.

        Para 'fpca' puro: viene del FPCA de skfda.
        Para encadenados: viene del PCA hecho sobre los coeficientes.
        """
        self._check_fitted()
        if "fpca" not in self.method:
            raise RuntimeError(
                f"fpca_explained_variance() no aplica a method='{self.method}'."
            )
        if self.method == "fpca":
            return np.asarray(self.fpca_.explained_variance_ratio_)
        return self._fpca_explained_variance_coefs

    def fpca_components(self, grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Componentes principales funcionales evaluadas en una grilla.

        Retorna
        -------
        comps : (K_fpca, G)
            Las K_fpca componentes en la grilla solicitada.
        """
        self._check_fitted()
        if "fpca" not in self.method:
            raise RuntimeError(
                f"fpca_components() no aplica a method='{self.method}'."
            )
        grid_used = self.grid_ if grid is None else np.asarray(grid, dtype=float)

        if self.method == "fpca":
            # Componentes funcionales directos de skfda
            comps_fd = self.fpca_.components_(grid_used)   # shape variable
            return np.asarray(comps_fd).squeeze().reshape(self.K_, -1)

        # Encadenado: cada componente es combinación lineal de phi_
        # comps_funcionales[k](t) = Σ_j eigvec[k,j] · phi_j(t)
        # → en grilla: comps[k, :] = eigvec[k, :] @ phi_
        return self._fpca_components_coefs @ self.phi_     # (K_fpca, G)

    # ─────────────────────────────────────────────────────────────────────
    # Constructor alternativo: from_coefficients (instancia "frozen")
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def from_coefficients(cls, THETA: np.ndarray) -> "FunctionalRepresentation":
        """
        Crea una instancia 'frozen' sin base subyacente, a partir de
        coeficientes ya proyectados externamente.

        Útil cuando el usuario tiene coeficientes calculados en R/fda u otro
        software y solo quiere usarlos como entrada al modelo PSBP.

        Parámetros
        ----------
        THETA : (T, K)

        Retorna
        -------
        instancia con `is_frozen_=True`. Métodos de reconstrucción, plots
        de bases y `transform` no estarán disponibles.
        """
        THETA = np.asarray(THETA, dtype=float)
        if THETA.ndim != 2:
            raise ValueError(f"THETA debe ser 2D; shape={THETA.shape}")
        obj = cls(method="bspline")  # placeholder
        obj.K_         = THETA.shape[1]
        obj.is_fitted_ = True
        obj.is_frozen_ = True
        obj._frozen_THETA = THETA
        return obj

    # ═════════════════════════════════════════════════════════════════════
    # Implementación interna
    # ═════════════════════════════════════════════════════════════════════

    # ── Cuadratura L² ────────────────────────────────────────────────────
    def _integrate_L2(self, F: np.ndarray,
                      grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Integra F sobre la grilla con la regla de cuadratura configurada.

        F: (T, G) o (G,). Retorna (T,) o escalar.
        """
        grid_used = self.grid_ if grid is None else grid
        if self.quadrature == "simpson":
            if len(grid_used) % 2 == 0:
                warnings.warn(
                    "Simpson requiere número impar de puntos. "
                    "Cayendo a trapezoidal para esta llamada.",
                    UserWarning,
                )
                return np.trapezoid(F, x=grid_used, axis=-1)
            return simpson(F, x=grid_used, axis=-1)
        return np.trapezoid(F, x=grid_used, axis=-1)

    # ── Etapa 1: B-spline ────────────────────────────────────────────────
    def _fit_bspline(self, grid: np.ndarray) -> None:
        try:
            from skfda.representation.basis import BSplineBasis
        except ImportError as e:
            raise ImportError(
                "skfda es requerido. Instalar con: pip install scikit-fda"
            ) from e

        basis = BSplineBasis(
            domain_range=self.domain_,
            n_basis=self.n_basis,
            order=self.order,
        )
        self.basis_ = basis
        phi = np.asarray(basis(grid)).squeeze()
        # Asegurar shape (K, G)
        if phi.shape[0] != self.n_basis:
            phi = phi.T
        self.phi_ = phi

        if self.projection == "l2":
            self._compute_gram()

    def _fit_fourier(self, grid: np.ndarray) -> None:
        try:
            from skfda.representation.basis import FourierBasis
        except ImportError as e:
            raise ImportError(
                "skfda es requerido. Instalar con: pip install scikit-fda"
            ) from e

        n_basis_used = self.n_basis
        if n_basis_used % 2 == 0:
            n_basis_used = n_basis_used + 1
            warnings.warn(
                f"FourierBasis requiere n_basis impar. "
                f"Ajustado a {n_basis_used}.",
                UserWarning,
            )

        period = self.domain_[1] - self.domain_[0]
        basis = FourierBasis(
            domain_range=self.domain_,
            n_basis=n_basis_used,
            period=period,
        )
        self.basis_ = basis
        phi = np.asarray(basis(grid)).squeeze()
        if phi.shape[0] != n_basis_used:
            phi = phi.T
        self.phi_ = phi

        if self.projection == "l2":
            self._compute_gram()

    def _compute_gram(self) -> None:
        """
        Matriz de Gram L²: G_jk = ∫ φ_j(t) φ_k(t) dt, con la cuadratura
        configurada.
        """
        K, G = self.phi_.shape
        # phi_outer[j, k, g] = phi_[j, g] * phi_[k, g]
        # Para evitar el array 3D (memoria), iteramos sobre k.
        gram = np.zeros((K, K))
        for k in range(K):
            integrand = self.phi_ * self.phi_[k][None, :]      # (K, G)
            gram[:, k] = self._integrate_L2(integrand)
        # Simetrizar por numérica
        self.gram_ = 0.5 * (gram + gram.T)

    # ── Proyección a la base inicial ─────────────────────────────────────
    def _project_to_base(self, Y: np.ndarray) -> np.ndarray:
        """
        Proyecta Y (T, G) sobre phi_ → coeficientes (T, K_base).
        """
        if self.projection == "discrete":
            # THETA' = (Φ Φ')⁻¹ Φ Y'  →  THETA = Y Φ' (Φ Φ')⁻¹
            A = self.phi_ @ self.phi_.T                   # (K, K)
            B = Y @ self.phi_.T                            # (T, K)
            return np.linalg.solve(A.T, B.T).T

        # L²: THETA = G⁻¹ B,  B_j = ∫ y(t) φ_j(t) dt
        # B[t, j] = ∫ Y[t,:] · phi_[j, :] dt
        # Vectorizado: B = ∫ Y[t,g] phi_[j,g] dg  → (T, K)
        B = np.zeros((Y.shape[0], self.phi_.shape[0]))
        for j in range(self.phi_.shape[0]):
            integrand = Y * self.phi_[j][None, :]          # (T, G)
            B[:, j] = self._integrate_L2(integrand)
        # Resolver gram_ @ THETA' = B'  → THETA = (gram⁻¹ B')'
        return np.linalg.solve(self.gram_, B.T).T

    # ── Etapa 2: FPCA directo sobre curvas ───────────────────────────────
    def _fit_fpca_direct(self, Y: np.ndarray, grid: np.ndarray) -> None:
        try:
            from skfda import FDataGrid
            from skfda.preprocessing.dim_reduction import FPCA
        except ImportError as e:
            raise ImportError(
                "skfda es requerido. Instalar con: pip install scikit-fda"
            ) from e

        fd = FDataGrid(data_matrix=Y, grid_points=grid,
                       domain_range=self.domain_)
        fpca = FPCA(n_components=self.n_basis_fpca)
        fpca.fit(fd)
        self.fpca_ = fpca
        # phi_ no aplica en FPCA directo; mantenemos None.

    def _transform_fpca_direct(self, Y: np.ndarray,
                               grid: np.ndarray) -> np.ndarray:
        from skfda import FDataGrid
        fd = FDataGrid(data_matrix=Y, grid_points=grid,
                       domain_range=self.domain_)
        return np.asarray(self.fpca_.transform(fd))

    def _reconstruct_fpca_direct(self, THETA: np.ndarray) -> np.ndarray:
        """
        Reconstrucción para 'fpca' puro: Y_hat = scores @ components + mean.
        """
        fpca = self.fpca_
        comps_grid = self.fpca_components(self.grid_)      # (K_fpca, G)
        mean_curve = np.asarray(fpca.mean_(self.grid_)).squeeze()
        return THETA @ comps_grid + mean_curve

    # ── Etapa 2: FPCA encadenado sobre coeficientes ──────────────────────
    def _fit_fpca_on_coefs(self, THETA_base: np.ndarray) -> None:
        """
        PCA estándar sobre los coeficientes de la base intermedia.

        THETA_base : (T, K_base)
        Genera:
            self._fpca_mean_coefs       : (K_base,)
            self._fpca_components_coefs : (K_fpca, K_base)
            self._fpca_explained_variance_coefs : (K_fpca,)
        """
        K_base = THETA_base.shape[1]
        if self.n_basis_fpca > K_base:
            raise ValueError(
                f"n_basis_fpca={self.n_basis_fpca} > n_basis={K_base}. "
                "El número de componentes FPCA no puede exceder K_base."
            )

        mean_c = THETA_base.mean(axis=0)
        Xc = THETA_base - mean_c                            # (T, K_base)

        # SVD: Xc = U S Vt → componentes son las filas de Vt
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        var_total = (S ** 2).sum()
        if var_total <= 0:
            raise ValueError(
                "Varianza nula en los coeficientes; no se puede aplicar FPCA."
            )

        self._fpca_mean_coefs            = mean_c
        self._fpca_components_coefs      = Vt[:self.n_basis_fpca, :]   # (K_fpca, K_base)
        self._fpca_explained_variance_coefs = (S ** 2 / var_total)[:self.n_basis_fpca]

    # ─────────────────────────────────────────────────────────────────────
    # Plots (mantenidos en la clase para uso del orquestador)
    # ─────────────────────────────────────────────────────────────────────
    def plot_basis(self, ax=None, title: Optional[str] = None,
                   alpha: float = 0.8, linewidth: float = 1.5):
        """Funciones de base (o FPCs) evaluadas en la grilla."""
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("plot_basis() no disponible para frozen.")

        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        if "fpca" in self.method:
            comps = self.fpca_components(self.grid_)       # (K_fpca, G)
            for k in range(self.K_):
                ax.plot(self.grid_, comps[k], alpha=alpha, lw=linewidth,
                        label=f"FPC {k+1}" if self.K_ <= 10 else None)
            ax.set_title(title or
                         f"FPCs | method='{self.method}' | K={self.K_}")
        else:
            for k in range(self.K_):
                ax.plot(self.grid_, self.phi_[k], alpha=alpha, lw=linewidth,
                        label=f"Base {k+1}" if self.K_ <= 10 else None)
            ax.set_title(title or
                         f"Base | method='{self.method}' | K={self.K_}")

        if self.K_ <= 10:
            ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("t")
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_curves(self, Y: np.ndarray, grid: Optional[np.ndarray] = None,
                    n_show: int = 5, ax=None,
                    title: Optional[str] = None, seed: Optional[int] = None):
        """Curvas originales vs reconstruidas."""
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("plot_curves() no disponible para frozen.")

        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        Y_arr = np.asarray(Y)
        grid_used = self.grid_ if grid is None else grid

        rng = np.random.default_rng(seed)
        T = Y_arr.shape[0]
        idx = rng.choice(T, size=min(n_show, T), replace=False)

        THETA = self.transform(Y_arr, grid_used)
        Y_hat = self.reconstruct(THETA)

        for i, k in enumerate(idx):
            color = f"C{i % 10}"
            ax.plot(grid_used, Y_arr[k], color=color, lw=1.2, alpha=0.7,
                    label="Original" if i == 0 else None)
            ax.plot(grid_used, Y_hat[k], color=color, lw=1.5, ls="--",
                    label="Reconstruida" if i == 0 else None)
        ax.set_xlabel("t"); ax.set_ylabel("Y(t)")
        ax.set_title(title or f"Originales vs reconstruidas (n={len(idx)})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_mean(self, Y: Optional[np.ndarray] = None, ax=None,
                  title: Optional[str] = None, show_ci: bool = True,
                  ci_alpha: float = 0.3):
        """Media funcional con banda ±1 std."""
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("plot_mean() no disponible para frozen.")

        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        if Y is not None:
            Y_arr = np.asarray(Y)
            mean_curve = Y_arr.mean(axis=0)
            std_curve = Y_arr.std(axis=0)
        else:
            mean_curve = self.mean_
            std_curve  = None

        ax.plot(self.grid_, mean_curve, color="darkblue", lw=2.5,
                label="Media funcional")
        if show_ci and std_curve is not None:
            ax.fill_between(self.grid_, mean_curve - std_curve,
                            mean_curve + std_curve,
                            color="darkblue", alpha=ci_alpha, label="±1 std")
        ax.set_xlabel("t"); ax.set_ylabel("Y(t)")
        ax.set_title(title or "Media funcional")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_coefficients(self, THETA: np.ndarray, ax=None,
                          title: Optional[str] = None,
                          cmap: str = "viridis"):
        """Heatmap (T, K) de coeficientes."""
        self._check_fitted()
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        im = ax.imshow(np.asarray(THETA), aspect="auto", cmap=cmap,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Coeficiente")
        ax.set_xlabel("k"); ax.set_ylabel("t (observación)")
        ax.set_title(title or
                     f"Coeficientes (T={THETA.shape[0]}, K={THETA.shape[1]})")
        return ax

    def plot_fpca_variance(self, ax=None, title: Optional[str] = None):
        """Scree plot de varianza explicada FPCA."""
        self._check_fitted()
        if "fpca" not in self.method:
            raise RuntimeError(
                f"plot_fpca_variance() no aplica a method='{self.method}'."
            )

        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        evr = self.fpca_explained_variance()
        cumsum = np.cumsum(evr)
        x = np.arange(1, len(evr) + 1)

        ax.bar(x, evr * 100, color="steelblue", alpha=0.7, label="Individual")
        ax.plot(x, cumsum * 100, color="crimson", marker="o", lw=2,
                label="Acumulada")
        ax.axhline(90, color="gray", ls="--", lw=1, label="90%")
        ax.set_xticks(x)
        ax.set_xlabel("FPC"); ax.set_ylabel("Varianza explicada (%)")
        ax.set_title(title or "Varianza explicada por FPCA")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    # ─────────────────────────────────────────────────────────────────────
    # Mini-reporte (combina plots para el orquestador)
    # ─────────────────────────────────────────────────────────────────────
    def report(self, Y: np.ndarray,
               grid: Optional[np.ndarray] = None,
               save_path: Optional[str] = None,
               n_curves_show: int = 8) -> Dict[str, Any]:
        """
        Genera figura de diagnóstico con 4 paneles (2x2):
            (a) Base / FPCs.
            (b) Media funcional ± 1 std.
            (c) Curvas originales vs reconstruidas.
            (d) Varianza explicada (si FPCA) o coeficientes (heatmap, si no).

        Retorna
        -------
        dict con métricas de reconstrucción y, si `save_path`, ruta de la figura.
        """
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("report() no disponible para frozen.")

        import matplotlib.pyplot as plt
        Y = np.asarray(Y, dtype=float)
        grid_used = self.grid_ if grid is None else np.asarray(grid, dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            f"FunctionalRepresentation — method='{self.method}' "
            f"| n_basis={self.n_basis}"
            + (f", n_basis_fpca={self.n_basis_fpca}" if "fpca" in self.method else "")
            + f" | projection='{self.projection}'",
            fontsize=11,
        )

        self.plot_basis(ax=axes[0, 0])
        self.plot_mean(Y=Y, ax=axes[0, 1])
        self.plot_curves(Y=Y, grid=grid_used, n_show=n_curves_show, ax=axes[1, 0])

        if "fpca" in self.method:
            self.plot_fpca_variance(ax=axes[1, 1])
        else:
            THETA = self.transform(Y, grid_used)
            self.plot_coefficients(THETA, ax=axes[1, 1])

        plt.tight_layout()
        result: Dict[str, Any] = {
            "reconstruction_error": self.reconstruction_error(Y, grid_used),
        }
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            result["save_path"] = save_path
        result["figure"] = fig
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Información y serialización
    # ─────────────────────────────────────────────────────────────────────
    def summary(self) -> None:
        if not self.is_fitted_:
            print(f"FunctionalRepresentation(method='{self.method}', "
                  f"not fitted)")
            return
        print(f"FunctionalRepresentation | method='{self.method}' | K={self.K_}")
        print(f"  n_basis      : {self.n_basis}")
        if "fpca" in self.method:
            print(f"  n_basis_fpca : {self.n_basis_fpca}")
        print(f"  projection   : {self.projection}")
        print(f"  quadrature   : {self.quadrature}")
        print(f"  center       : {self.center}")
        if self.grid_ is not None:
            print(f"  grid         : {len(self.grid_)} pts en "
                  f"[{self.grid_.min():.3f}, {self.grid_.max():.3f}]")
        if "fpca" in self.method:
            try:
                evr = self.fpca_explained_variance()
                print(f"  var. acumul. : {np.cumsum(evr)[-1]*100:.2f}%")
            except Exception:
                pass

    def get_config(self) -> Dict[str, Any]:
        """Configuración serializable (no incluye objetos skfda)."""
        return {
            "method":        self.method,
            "n_basis":       self.n_basis,
            "n_basis_fpca":  self.n_basis_fpca,
            "order":         self.order,
            "domain":        self.domain,
            "projection":    self.projection,
            "quadrature":    self.quadrature,
            "center":        self.center,
            "K_":            self.K_,
            "is_fitted":     self.is_fitted_,
            "is_frozen":     self.is_frozen_,
        }

    def __repr__(self) -> str:
        status = (f"frozen(K={self.K_})" if self.is_frozen_
                  else (f"fitted(K={self.K_})" if self.is_fitted_ else "not fitted"))
        return (f"FunctionalRepresentation(method='{self.method}', "
                f"n_basis={self.n_basis}, "
                + (f"n_basis_fpca={self.n_basis_fpca}, "
                   if "fpca" in self.method else "")
                + f"{status})")

    # ─────────────────────────────────────────────────────────────────────
    # Validaciones internas
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_Y_grid(Y: np.ndarray, grid: np.ndarray) -> None:
        if Y.ndim != 2:
            raise ValueError(f"Y debe ser 2D (T, G); recibido {Y.shape}")
        if grid.ndim != 1:
            raise ValueError(f"grid debe ser 1D (G,); recibido {grid.shape}")
        if Y.shape[1] != len(grid):
            raise ValueError(
                f"Y tiene {Y.shape[1]} columnas pero grid tiene {len(grid)} puntos"
            )
        if not np.all(np.diff(grid) > 0):
            raise ValueError("grid debe ser estrictamente creciente.")

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "Objeto no ajustado. Llame a fit() o fit_transform() primero."
            )
