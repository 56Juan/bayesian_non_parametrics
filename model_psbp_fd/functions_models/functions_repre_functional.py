"""
functions_repre_functional.py
=============================

Representacion funcional de curvas para el pipeline PSBP-FD.

Pipeline conceptual:
    curvas Y (T, G) --> FunctionalRepresentation --> coeficientes THETA (T, K)
    (T = observaciones, G = puntos de grilla, K = coeficientes de base)

Metodos soportados:
    - "bspline" : proyeccion sobre base B-spline.
    - "fourier" : proyeccion sobre base de Fourier.

Esta clase es exclusivamente un SUAVIZADOR: proyecta curvas discretas sobre un
sistema de base y las reconstruye. La reduccion de dimension por componentes
principales NO vive aqui: el FPCA del proyecto es el generalizado en metrica
L^2 (problema propio C u = lambda W u, con W la matriz de Gram de la base) e
implementado en `functions_models.functions_fpca.FPCA_L2`, que opera sobre los
coeficientes THETA que esta clase produce. Las variantes encadenadas
('bspline+fpca', 'fourier+fpca') de versiones anteriores aplicaban PCA
euclideo sin la metrica de Gram ---incorrecto para bases no ortonormales--- y
fueron eliminadas.

Proyeccion sobre bases:
    - "l2"       : proyeccion en L^2 con cuadratura trapezoidal.
                   THETA = G^{-1} B,  G_jk = int phi_j phi_k dt,
                   B_j = int y phi_j dt.
    - "discrete" : proyeccion discreta (ignora el espaciamiento de la grilla).
                   THETA = (Phi Phi')^{-1} Phi Y'.

Cuadratura: se emplea UNICAMENTE la regla trapezoidal, que es la regla comun a
todo el proyecto (`utils.quadrature`). Mezclar reglas entre el suavizado y la
FPCA degrada la ortonormalidad de las autofunciones sin producir error alguno,
por lo que la opcion de Simpson de versiones anteriores fue eliminada.

Error de reconstruccion en dos normas (siempre devuelve ambas):
    - L^2 integrada : ||y - y_hat||_L2 = sqrt(int (y(t) - y_hat(t))^2 dt).
    - Punto a punto : RMSE sobre la grilla.

Convencion de "frozen": instancias creadas via `from_coefficients` no tienen
base subyacente; solo exponen los coeficientes y la informacion minima.
Operaciones de reconstruccion y graficos de base no estan disponibles.

Author: model_psbp_fd
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any, Optional, Dict, Tuple
import warnings

import numpy as np

from ..utils.quadrature import pesos_trapezoidales


# ─────────────────────────────────────────────────────────────────────────────
# Tipos
# ─────────────────────────────────────────────────────────────────────────────

RepreMethod      = Literal["bspline", "fourier"]
ProjectionMethod = Literal["l2", "discrete"]


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionalRepresentation:
    """
    Representacion funcional de curvas discretas para PSBP-FD.

    Parametros
    ----------
    method : {'bspline', 'fourier'}
        Sistema de base de la proyeccion.

    n_basis : int, default=20
        Numero de funciones de base (K final). Para 'fourier' se ajusta al
        impar superior si se entrega par.

    order : int, default=4
        Orden del spline (4 = cubico). Solo aplica a 'bspline'.

    domain : tuple[float, float], opcional
        Dominio de las curvas. Si None, se infiere de `grid` en `fit`.

    projection : {'l2', 'discrete'}, default='l2'
        Tipo de proyeccion. La proyeccion 'l2' emplea la cuadratura
        trapezoidal comun del proyecto.

    center : bool, default=False
        Si True, centra las curvas (resta la media funcional del bloque de
        ajuste) antes de proyectar, y la vuelve a sumar en `reconstruct`.

    Atributos post-fit
    ------------------
    grid_      : (G,)     grilla de evaluacion
    domain_    : (a, b)   dominio efectivo
    K_         : int      dimension de los coeficientes
    mean_      : (G,)     media funcional de las curvas de fit
    phi_       : (K, G)   base evaluada en la grilla
    gram_      : (K, K)   matriz de Gram L^2 (solo si projection='l2')
    w_quad_    : (G,)     pesos de la cuadratura trapezoidal
    basis_     : objeto skfda de la base
    is_fitted_ : bool
    """

    # ── Configuracion (input del usuario) ────────────────────────────────────
    method:     RepreMethod      = "bspline"
    n_basis:    int              = 20
    order:      int              = 4
    domain:     Optional[Tuple[float, float]] = None
    projection: ProjectionMethod = "l2"
    center:     bool             = False

    # ── Estado interno (post-fit) ────────────────────────────────────────────
    grid_:      Optional[np.ndarray] = field(default=None, repr=False)
    domain_:    Optional[Tuple[float, float]] = field(default=None, repr=False)
    K_:         Optional[int]        = field(default=None, repr=False)
    mean_:      Optional[np.ndarray] = field(default=None, repr=False)
    phi_:       Optional[np.ndarray] = field(default=None, repr=False)
    gram_:      Optional[np.ndarray] = field(default=None, repr=False)
    w_quad_:    Optional[np.ndarray] = field(default=None, repr=False)
    basis_:     Any                  = field(default=None, repr=False)
    is_fitted_: bool                 = field(default=False, repr=False)
    is_frozen_: bool                 = field(default=False, repr=False)

    # ─────────────────────────────────────────────────────────────────────
    # Validacion de configuracion (al construir)
    # ─────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        valid_methods = {"bspline", "fourier"}
        if self.method not in valid_methods:
            extra = ""
            if isinstance(self.method, str) and "fpca" in self.method:
                extra = (
                    " Las variantes con FPCA interna fueron eliminadas: la "
                    "FPCA del proyecto es el problema propio generalizado en "
                    "metrica L^2 y vive en functions_models.FPCA_L2, aplicado "
                    "sobre los coeficientes que esta clase produce."
                )
            raise ValueError(
                f"method='{self.method}' invalido. Opciones: "
                f"{sorted(valid_methods)}.{extra}"
            )
        if self.projection not in {"l2", "discrete"}:
            raise ValueError(
                f"projection='{self.projection}' invalido. Use 'l2' o 'discrete'."
            )
        if self.n_basis < 2:
            raise ValueError(f"n_basis debe ser >= 2; recibido {self.n_basis}")
        if self.method == "bspline" and self.order < 1:
            raise ValueError(f"order debe ser >= 1; recibido {self.order}")

    # ─────────────────────────────────────────────────────────────────────
    # API principal: fit / transform / reconstruct
    # ─────────────────────────────────────────────────────────────────────
    def fit(self, Y: np.ndarray, grid: np.ndarray) -> "FunctionalRepresentation":
        """
        Ajusta la base sobre la grilla y calcula la media funcional del
        bloque de ajuste.

        Bajo el esquema de retencion temporal, `Y` debe contener unicamente
        las curvas del bloque de entrenamiento; `transform` puede aplicarse
        despues a cualquier bloque.
        """
        Y = np.asarray(Y, dtype=float)
        grid = np.asarray(grid, dtype=float)
        self._validate_Y_grid(Y, grid)

        self.grid_ = grid
        self.domain_ = self.domain or (float(grid.min()), float(grid.max()))
        self.mean_ = Y.mean(axis=0)
        self.w_quad_ = pesos_trapezoidales(grid)

        if self.method == "bspline":
            self._fit_bspline(grid)
        elif self.method == "fourier":
            self._fit_fourier(grid)
        else:  # pragma: no cover - bloqueado en __post_init__
            raise ValueError(f"method='{self.method}' no implementado.")

        self.K_ = self.phi_.shape[0]
        self.is_fitted_ = True
        return self

    def transform(self, Y: np.ndarray,
                  grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Proyecta nuevas curvas en la base ajustada -> THETA (T, K).

        Parametros
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
                "Instancia frozen (creada via from_coefficients). "
                "transform() no aplica; los coeficientes ya son la representacion."
            )

        Y = np.asarray(Y, dtype=float)
        grid_used = self.grid_ if grid is None else np.asarray(grid, dtype=float)
        if grid is not None and not np.allclose(grid_used, self.grid_):
            warnings.warn(
                "La grilla entregada difiere de la usada en fit(); la base "
                "evaluada corresponde a la grilla original.",
                UserWarning,
            )

        if Y.ndim != 2 or Y.shape[1] != len(grid_used):
            raise ValueError(
                f"Y debe ser (T, G={len(grid_used)}); recibido {Y.shape}"
            )

        Y_work = (Y - self.mean_) if self.center else Y
        return self._project_to_base(Y_work)

    def fit_transform(self, Y: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Atajo: fit(Y, grid).transform(Y, grid)."""
        return self.fit(Y, grid).transform(Y, grid)

    # ─────────────────────────────────────────────────────────────────────
    # Reconstruccion y error
    # ─────────────────────────────────────────────────────────────────────
    def reconstruct(self, THETA: np.ndarray) -> np.ndarray:
        """
        Reconstruye curvas en escala original a partir de los coeficientes.

        Si `center=True` fue usado en fit, la media funcional se vuelve a
        sumar automaticamente.

        Parametros
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
        Y_hat = THETA @ self.phi_                          # (T, G)

        if self.center:
            Y_hat = Y_hat + self.mean_
        return Y_hat

    def reconstruction_error(self,
                             Y: np.ndarray,
                             grid: Optional[np.ndarray] = None
                             ) -> Dict[str, float]:
        """
        Error de reconstruccion en dos normas:

            - 'l2'       : ||y - y_hat||_L2 = sqrt(int (y(t) - y_hat(t))^2 dt)
            - 'pointwise': RMSE punto a punto sobre la grilla

        Retorna
        -------
        dict con:
            'rmse_l2_mean',  'rmse_l2_std'
            'rmse_pw_mean',  'rmse_pw_std'
            'rel_l2_mean',   'rel_l2_std'
        """
        self._check_fitted()
        Y = np.asarray(Y, dtype=float)
        grid_used = self.grid_ if grid is None else np.asarray(grid, dtype=float)

        THETA = self.transform(Y, grid_used)
        Y_hat = self.reconstruct(THETA)
        residuals = Y - Y_hat                              # (T, G)

        # L^2 integrada (cuadratura trapezoidal comun)
        l2_sq = self._integrate_L2(residuals ** 2)         # (T,)
        l2_norm_y = np.sqrt(self._integrate_L2(Y ** 2))
        rmse_l2 = np.sqrt(l2_sq)
        rel_l2 = rmse_l2 / (l2_norm_y + 1e-12)

        # Punto a punto
        rmse_pw = np.sqrt((residuals ** 2).mean(axis=1))

        return {
            "rmse_l2_mean": float(rmse_l2.mean()),
            "rmse_l2_std":  float(rmse_l2.std()),
            "rmse_pw_mean": float(rmse_pw.mean()),
            "rmse_pw_std":  float(rmse_pw.std()),
            "rel_l2_mean":  float(rel_l2.mean()),
            "rel_l2_std":   float(rel_l2.std()),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Constructor alternativo: from_coefficients (instancia "frozen")
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def from_coefficients(cls, THETA: np.ndarray) -> "FunctionalRepresentation":
        """
        Crea una instancia 'frozen' sin base subyacente, a partir de
        coeficientes ya proyectados externamente.

        Util cuando el usuario tiene coeficientes calculados en R/fda u otro
        software y solo quiere usarlos como entrada al modelo PSBP.

        Parametros
        ----------
        THETA : (T, K)

        Retorna
        -------
        instancia con `is_frozen_=True`. Metodos de reconstruccion, graficos
        de bases y `transform` no estaran disponibles.
        """
        THETA = np.asarray(THETA, dtype=float)
        if THETA.ndim != 2:
            raise ValueError(f"THETA debe ser 2D; shape={THETA.shape}")
        obj = cls(method="bspline")  # placeholder
        obj.K_ = THETA.shape[1]
        obj.is_fitted_ = True
        obj.is_frozen_ = True
        obj._frozen_THETA = THETA
        return obj

    # ═════════════════════════════════════════════════════════════════════
    # Implementacion interna
    # ═════════════════════════════════════════════════════════════════════

    # ── Cuadratura L^2 ───────────────────────────────────────────────────
    def _integrate_L2(self, F: np.ndarray) -> np.ndarray:
        """
        Integra F (T, G) o (G,) sobre la grilla con la cuadratura trapezoidal
        del proyecto. F @ w equivale a np.trapezoid(F, x=grid) sobre la misma
        grilla, pero deja explicitos los pesos que comparten la matriz de
        Gram, la proyeccion y la FPCA generalizada.
        """
        return np.asarray(F, dtype=float) @ self.w_quad_

    # ── Ajuste de bases ──────────────────────────────────────────────────
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
        self.phi_ = self._eval_basis(basis, grid, self.n_basis)

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
        self.phi_ = self._eval_basis(basis, grid, n_basis_used)

        if self.projection == "l2":
            self._compute_gram()

    @staticmethod
    def _eval_basis(basis, grid: np.ndarray, n_basis: int) -> np.ndarray:
        """
        Evalua la base en la grilla garantizando la forma (K, G).

        La orientacion se decide contra la longitud de la grilla y no contra
        `n_basis`: cuando K == G ambas dimensiones coinciden y una comparacion
        contra `n_basis` no puede detectar una matriz transpuesta. skfda
        retorna la evaluacion con las funciones en el primer eje, de modo que
        en el caso cuadrado la forma ya es la correcta y no debe transponerse.
        """
        phi = np.asarray(basis(grid)).squeeze()
        G = len(grid)
        if phi.ndim != 2:
            raise RuntimeError(
                f"La evaluacion de la base retorno forma {phi.shape}; se "
                "esperaba una matriz 2D."
            )
        if phi.shape == (n_basis, G):
            return phi
        if phi.shape == (G, n_basis):
            return phi.T
        raise RuntimeError(
            f"La evaluacion de la base tiene forma {phi.shape}, incompatible "
            f"con (K={n_basis}, G={G})."
        )

    # ── Matriz de Gram y proyeccion ──────────────────────────────────────
    def _compute_gram(self) -> None:
        """
        Matriz de Gram L^2: G_jk = int phi_j(t) phi_k(t) dt, con la cuadratura
        trapezoidal del proyecto, vectorizada:

            gram = (phi * w) phi',

        y simetrizada para eliminar la asimetria numerica residual.
        """
        Phi_w = self.phi_ * self.w_quad_[None, :]          # (K, G)
        gram = Phi_w @ self.phi_.T                          # (K, K)
        self.gram_ = 0.5 * (gram + gram.T)

    def _project_to_base(self, Y: np.ndarray) -> np.ndarray:
        """
        Proyecta Y (T, G) sobre phi_ -> coeficientes (T, K).

        Proyeccion 'l2' vectorizada:
            B[t, j] = int y_t(s) phi_j(s) ds = (Y * w) phi'[:, j]
            THETA   = solve(gram, B')'.
        """
        if self.projection == "discrete":
            # THETA' = (Phi Phi')^{-1} Phi Y'  ->  THETA = Y Phi' (Phi Phi')^{-1}
            A = self.phi_ @ self.phi_.T                     # (K, K)
            B = Y @ self.phi_.T                             # (T, K)
            return np.linalg.solve(A.T, B.T).T

        B = (Y * self.w_quad_[None, :]) @ self.phi_.T       # (T, K)
        return np.linalg.solve(self.gram_, B.T).T

    # ─────────────────────────────────────────────────────────────────────
    # Graficos (mantenidos en la clase para uso del orquestador)
    # ─────────────────────────────────────────────────────────────────────
    def plot_basis(self, ax=None, title: Optional[str] = None,
                   alpha: float = 0.8, linewidth: float = 1.5):
        """Funciones de base evaluadas en la grilla."""
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("plot_basis() no disponible para frozen.")

        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

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
        """Curvas originales vs reconstruidas (muestra aleatoria)."""
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("plot_curves() no disponible para frozen.")

        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        Y_arr = np.asarray(Y, dtype=float)
        grid_used = self.grid_ if grid is None else np.asarray(grid, dtype=float)
        rng = np.random.default_rng(seed)
        idx = rng.choice(Y_arr.shape[0], size=min(n_show, Y_arr.shape[0]),
                         replace=False)

        THETA = self.transform(Y_arr[idx], grid_used)
        Y_hat = self.reconstruct(THETA)

        colors = plt.cm.tab10(np.linspace(0, 1, len(idx)))
        for k, (i, color) in enumerate(zip(idx, colors)):
            ax.plot(grid_used, Y_arr[i], color=color, lw=1.2, alpha=0.7,
                    label=f"obs {i}" if len(idx) <= 8 else None)
            ax.plot(grid_used, Y_hat[k], color=color, lw=1.5, ls="--",
                    alpha=0.9)
        ax.set_title(title or
                     f"Original (solida) vs reconstruida (punteada) | K={self.K_}")
        if len(idx) <= 8:
            ax.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("t")
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_mean(self, Y: Optional[np.ndarray] = None, ax=None,
                  title: Optional[str] = None):
        """Media funcional +- 1 desviacion estandar puntual."""
        self._check_fitted()
        if self.is_frozen_:
            raise RuntimeError("plot_mean() no disponible para frozen.")

        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        if Y is not None:
            Y_arr = np.asarray(Y, dtype=float)
            mean_curve = Y_arr.mean(axis=0)
            std_curve = Y_arr.std(axis=0)
            ax.fill_between(self.grid_, mean_curve - std_curve,
                            mean_curve + std_curve, alpha=0.25,
                            label="+- 1 std")
        else:
            mean_curve = self.mean_
        ax.plot(self.grid_, mean_curve, color="darkblue", lw=2.5,
                label="media")
        ax.set_title(title or "Media funcional")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("t")
        ax.grid(True, ls="--", alpha=0.4)
        return ax

    def plot_coefficients(self, THETA: np.ndarray, ax=None,
                          title: Optional[str] = None):
        """Mapa de calor de los coeficientes (T x K)."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        im = ax.imshow(np.asarray(THETA, dtype=float).T, aspect="auto",
                       origin="lower", cmap="RdBu_r")
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xlabel("t (observacion)")
        ax.set_ylabel("coeficiente k")
        ax.set_title(title or f"Coeficientes THETA' ({self.K_} x T)")
        return ax

    def report(self, Y: np.ndarray, grid: Optional[np.ndarray] = None,
               n_curves_show: int = 5,
               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Genera figura de diagnostico con 4 paneles (2x2):
            (a) Base.
            (b) Media funcional +- 1 std.
            (c) Curvas originales vs reconstruidas.
            (d) Coeficientes (mapa de calor).

        Retorna
        -------
        dict con metricas de reconstruccion y, si `save_path`, ruta de la figura.
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
            f" | projection='{self.projection}'",
            fontsize=11,
        )

        self.plot_basis(ax=axes[0, 0])
        self.plot_mean(Y=Y, ax=axes[0, 1])
        self.plot_curves(Y=Y, grid=grid_used, n_show=n_curves_show, ax=axes[1, 0])

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
    # Informacion y serializacion
    # ─────────────────────────────────────────────────────────────────────
    def summary(self) -> None:
        if not self.is_fitted_:
            print(f"FunctionalRepresentation(method='{self.method}', "
                  f"not fitted)")
            return
        print(f"FunctionalRepresentation | method='{self.method}' | K={self.K_}")
        print(f"  n_basis      : {self.n_basis}")
        print(f"  projection   : {self.projection}")
        print(f"  quadrature   : trapezoidal (regla comun del proyecto)")
        print(f"  center       : {self.center}")
        if self.grid_ is not None:
            print(f"  grid         : {len(self.grid_)} pts en "
                  f"[{self.grid_.min():.3f}, {self.grid_.max():.3f}]")

    def get_config(self) -> Dict[str, Any]:
        """Configuracion serializable (no incluye objetos skfda)."""
        return {
            "method":     self.method,
            "n_basis":    self.n_basis,
            "order":      self.order,
            "domain":     self.domain,
            "projection": self.projection,
            "quadrature": "trapezoidal",
            "center":     self.center,
            "K_":         self.K_,
            "is_fitted":  self.is_fitted_,
            "is_frozen":  self.is_frozen_,
        }

    def __repr__(self) -> str:
        status = (f"frozen(K={self.K_})" if self.is_frozen_
                  else (f"fitted(K={self.K_})" if self.is_fitted_ else "not fitted"))
        return (f"FunctionalRepresentation(method='{self.method}', "
                f"n_basis={self.n_basis}, {status})")

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
