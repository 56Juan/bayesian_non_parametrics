"""
functional_representation_extended.py
========================================
Clase de representación funcional.
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
    mean_     : np.ndarray (G,)     media funcional de los datos de ajuste
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
    mean_:     np.ndarray | None = field(default=None, repr=False)
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

            # Media funcional de los datos de ajuste
            self.mean_ = Y.mean(axis=0)

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
            G_fit = self.phi_.shape[1]
            Y_arr = np.asarray(Y)
            if Y_arr.ndim != 2:
                raise ValueError(
                    f"Y debe ser 2D (T, G). Shape recibido: {Y_arr.shape}."
                )
            if Y_arr.shape[1] != G_fit:
                raise ValueError(
                    f"G inconsistente: ajustado con G={G_fit}, "
                    f"recibido G={Y_arr.shape[1]}."
                )
            phi   = self.phi_                          # (K, G)
            A     = phi @ phi.T                        # (K, K)
            B     = Y_arr @ phi.T                      # (T, K)
            THETA = np.linalg.solve(A, B.T).T          # (T, K)
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

        if self.method == "precomputed":
            raise RuntimeError(
                "reconstruction_error() no disponible para method='precomputed': "
                "no hay base funcional almacenada con la que reconstruir."
            )

        THETA = self.transform(Y, grid)
        Y_hat = self.reconstruct(THETA)
        residuals = Y - Y_hat
        rmse_per_curve = np.sqrt((residuals ** 2).mean(axis=1))
        norm_per_curve = np.sqrt((Y ** 2).mean(axis=1))
        return {
            "rmse_mean":      float(rmse_per_curve.mean()),
            "rmse_std":       float(rmse_per_curve.std()),
            "rel_error_mean": float((rmse_per_curve / (norm_per_curve + 1e-12)).mean()),
        }

    # ── Media funcional ───────────────────────────────────────────────

    def mean_functional(self, Y: np.ndarray | None = None) -> np.ndarray:
        """
        Devuelve la media funcional.

        Si Y es None, retorna la media calculada durante fit().
        Si Y se proporciona, calcula la media sobre ese array (T, G).

        Returns
        -------
        mean_curve : np.ndarray (G,)
        """
        self._check_fitted()

        if self.method == "precomputed":
            raise RuntimeError(
                "mean_functional() no disponible para method='precomputed': "
                "no hay grilla funcional definida."
            )

        if Y is None:
            if self.mean_ is None:
                raise RuntimeError(
                    "No hay media almacenada. Proporcione Y o re-ajuste el modelo."
                )
            return self.mean_

        Y_arr = np.asarray(Y)
        if Y_arr.ndim != 2:
            raise ValueError(f"Y debe ser 2D (T, G). Shape: {Y_arr.shape}.")
        return Y_arr.mean(axis=0)

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

        phi_sq    = np.squeeze(basis(grid))
        self.phi_ = phi_sq if phi_sq.shape[0] == self.n_basis else phi_sq.T
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

        n_basis_used = self.n_basis
        if n_basis_used % 2 == 0:
            n_basis_used = n_basis_used + 1
            import warnings
            warnings.warn(
                f"FourierBasis requiere n_basis impar. "
                f"n_basis={self.n_basis} ajustado a {n_basis_used}.",
                UserWarning,
                stacklevel=4,
            )

        period = domain[1] - domain[0]
        basis  = FourierBasis(
            domain_range=domain,
            n_basis=n_basis_used,
            period=period,
        )
        self.basis_ = basis

        phi_sq    = np.squeeze(basis(grid))
        phi_      = phi_sq if phi_sq.shape[0] <= phi_sq.shape[1] else phi_sq.T
        self.phi_ = phi_
        self.K_   = phi_.shape[0]

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
        self.phi_   = None
        self.K_     = self.n_basis

    def _transform_fpca(self, Y: np.ndarray, grid: np.ndarray) -> np.ndarray:
        from skfda import FDataGrid
        domain = self.domain or (float(grid.min()), float(grid.max()))
        fd     = FDataGrid(data_matrix=Y, grid_points=grid, domain_range=domain)
        return self.basis_.transform(fd)

    def _reconstruct_fpca(self, THETA: np.ndarray) -> np.ndarray:
        fpca    = self.basis_
        grid    = self.grid_
        G       = len(grid)
        K       = self.K_

        comps = np.zeros((K, G))
        for k in range(K):
            comps[k] = fpca.components_(grid)[k].squeeze()

        if self.center_fpca and hasattr(fpca, "mean_"):
            mean_curve = fpca.mean_(grid).squeeze()
        else:
            mean_curve = np.zeros(G)

        return THETA @ comps + mean_curve

    # ── Método "precomputed" ──────────────────────────────────────────

    def _fit_precomputed(self, THETA: np.ndarray) -> None:
        if not isinstance(THETA, np.ndarray) or THETA.ndim != 2:
            raise ValueError(
                "Para method='precomputed', Y debe ser np.ndarray 2D (T, K)."
            )
        self.K_    = THETA.shape[1]
        self.grid_ = None
        self.phi_  = None
        self.mean_ = None

    # ── Gráficos complementarios ──────────────────────────────────────

    def plot_basis(
        self,
        ax=None,
        title: str | None = None,
        alpha: float = 0.8,
        linewidth: float = 1.5,
    ):
        """
        Grafica las funciones de base evaluadas en la grilla.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        title : str, optional
        alpha : float
        linewidth : float

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        self._check_fitted()
        if self.method == "precomputed":
            raise RuntimeError("plot_basis() no disponible para 'precomputed'.")

        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        grid = self.grid_
        if self.method in ("bspline", "fourier"):
            for k in range(self.K_):
                ax.plot(grid, self.phi_[k], alpha=alpha, lw=linewidth,
                        label=f"Base {k+1}" if self.K_ <= 10 else None)
            if self.K_ <= 10:
                ax.legend(loc="upper right", fontsize=7)
        elif self.method == "fpca":
            fpca = self.basis_
            for k in range(self.K_):
                ax.plot(grid, fpca.components_(grid)[k].squeeze(),
                        alpha=alpha, lw=linewidth,
                        label=f"FPC {k+1}" if self.K_ <= 10 else None)
            if self.K_ <= 10:
                ax.legend(loc="upper right", fontsize=7)

        ax.set_xlabel("t")
        ax.set_ylabel("Base / Componente")
        ax.set_title(title or f"Funciones de base | method='{self.method}' | K={self.K_}")
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_curves(
        self,
        Y: np.ndarray,
        grid: np.ndarray | None = None,
        n_show: int = 5,
        ax=None,
        title: str | None = None,
    ):
        """
        Grafica curvas originales vs reconstruidas.

        Parameters
        ----------
        Y : np.ndarray (T, G)
        grid : np.ndarray (G,), optional
        n_show : int
            Número de curvas a mostrar (elegidas aleatoriamente si T > n_show).
        ax : matplotlib.axes.Axes, optional
        title : str, optional

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        self._check_fitted()
        if self.method == "precomputed":
            raise RuntimeError("plot_curves() no disponible para 'precomputed'.")

        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        Y_arr = np.asarray(Y)
        T = Y_arr.shape[0]
        grid_used = grid if grid is not None else self.grid_

        idx = np.random.choice(T, size=min(n_show, T), replace=False)
        THETA = self.transform(Y_arr, grid_used)
        Y_hat = self.reconstruct(THETA)

        for i, k in enumerate(idx):
            color = f"C{i % 10}"
            ax.plot(grid_used, Y_arr[k], color=color, lw=1.2, alpha=0.7,
                    label=f"Original {k}" if i == 0 else None)
            ax.plot(grid_used, Y_hat[k], color=color, lw=1.5, ls="--",
                    label=f"Reconstruida {k}" if i == 0 else None)

        ax.set_xlabel("t")
        ax.set_ylabel("Y(t)")
        ax.set_title(title or f"Curvas originales vs reconstruidas (n={len(idx)})")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_coefficients(
        self,
        THETA: np.ndarray,
        ax=None,
        title: str | None = None,
        cmap: str = "viridis",
    ):
        """
        Heatmap de coeficientes THETA (T, K).

        Parameters
        ----------
        THETA : np.ndarray (T, K)
        ax : matplotlib.axes.Axes, optional
        title : str, optional
        cmap : str

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        self._check_fitted()
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        im = ax.imshow(np.asarray(THETA), aspect="auto", cmap=cmap,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Valor del coeficiente")
        ax.set_xlabel("k (índice de coeficiente)")
        ax.set_ylabel("t (observación)")
        ax.set_title(title or f"Coeficientes THETA (T={THETA.shape[0]}, K={THETA.shape[1]})")
        return ax

    def plot_fpca_variance(self, ax=None, title: str | None = None):
        """
        Scree plot de varianza explicada por FPCA.
        Solo disponible para method='fpca'.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        self._check_fitted()
        if self.method != "fpca":
            raise RuntimeError("plot_fpca_variance() solo disponible para method='fpca'.")

        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        evr = self.fpca_explained_variance()
        cumsum = np.cumsum(evr)
        x = np.arange(1, len(evr) + 1)

        ax.bar(x, evr * 100, color="steelblue", alpha=0.7, label="Individual")
        ax.plot(x, cumsum * 100, color="crimson", marker="o", lw=2,
                label="Acumulada")
        ax.axhline(90, color="gray", ls="--", lw=1, label="90%")
        ax.set_xticks(x)
        ax.set_xlabel("Componente principal funcional")
        ax.set_ylabel("Varianza explicada (%)")
        ax.set_title(title or "Varianza explicada por FPCA")
        ax.legend(loc="best")
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_mean(
        self,
        Y: np.ndarray | None = None,
        ax=None,
        title: str | None = None,
        show_ci: bool = True,
        ci_alpha: float = 0.3,
    ):
        """
        Grafica la media funcional con banda de desviación estándar opcional.

        Parameters
        ----------
        Y : np.ndarray (T, G), optional
            Si se proporciona, calcula media/std sobre Y.
            Si es None, usa la media almacenada en fit().
        ax : matplotlib.axes.Axes, optional
        title : str, optional
        show_ci : bool
            Muestra banda de ±1 desviación estándar.
        ci_alpha : float
            Transparencia de la banda.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        self._check_fitted()
        if self.method == "precomputed":
            raise RuntimeError("plot_mean() no disponible para 'precomputed'.")

        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        grid = self.grid_

        if Y is not None:
            Y_arr = np.asarray(Y)
            mean_curve = Y_arr.mean(axis=0)
            std_curve = Y_arr.std(axis=0)
        else:
            mean_curve = self.mean_
            if mean_curve is None:
                raise RuntimeError("No hay media almacenada. Proporcione Y.")
            std_curve = None

        ax.plot(grid, mean_curve, color="darkblue", lw=2.5, label="Media funcional")

        if show_ci and std_curve is not None:
            ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve,
                            color="darkblue", alpha=ci_alpha, label="±1 std")

        ax.set_xlabel("t")
        ax.set_ylabel("Y(t)")
        ax.set_title(title or "Media funcional")
        ax.legend(loc="best")
        ax.grid(True, ls="--", alpha=0.4)
        return ax

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
            "mean_":       self.mean_.copy() if self.mean_ is not None else None,
        }

    def summary(self) -> None:
        self._check_fitted()
        print(f"FunctionalRepresentation | method='{self.method}' | "
              f"n_basis={self.n_basis} | K={self.K_}")
        if self.grid_ is not None:
            print(f"  grid: {len(self.grid_)} puntos en "
                  f"[{self.grid_.min():.3f}, {self.grid_.max():.3f}]")
        if self.mean_ is not None:
            print(f"  media funcional: almacenada (G={len(self.mean_)})")
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