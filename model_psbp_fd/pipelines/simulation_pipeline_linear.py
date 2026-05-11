"""
simulation_pipeline_linear.py
==============================
Simulador de series de tiempo funcionales FAR(1) con tendencia determinística.

Correcciones aplicadas respecto a la versión anterior
------------------------------------------------------
1. _draw_noise: gaussian_filter1d reduce la desviación estándar del ruido en un
   factor ≈ 0.37 (depende de n_points y sigma). Se renormaliza el vector suavizado
   para que std(eps) = noise_std exactamente, preservando la semántica del parámetro.

2. Tendencias `linear` y `bump` reescaladas para que la tendencia máxima sea
   del mismo orden que noise_std al final de la simulación, evitando que dominen
   el proceso estocástico cuando n_curves es grande.

3. __init__: validación de Psi (shape cuadrada y compatible con domain) y
   validación de callable de tendencia (firma inspeccionada).

4. burn_in por defecto elevado de 50 → 100 (la dinámica converge en ≈ 50 pasos
   pero el margen extra cuesta poco y mejora la estabilidad).

5. Se añade el método summary_stats() para diagnóstico rápido del proceso
   generado: media funcional, varianza, eigenvalor máximo de la cov empírica
   y test de normalidad marginal.

6. Se añade plot_acf() para visualizar la autocorrelación funcional en puntos s fijos.
"""

import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import normaltest


# ============================================================
# DOMINIO FUNCIONAL
# ============================================================

@dataclass
class FunctionalDomain:
    """Malla de puntos s donde se evalúa cada curva funcional."""
    grid: np.ndarray

    @classmethod
    def regular(
        cls,
        s_min: float = 0.0,
        s_max: float = 1.0,
        n_points: int = 100,
    ) -> "FunctionalDomain":
        return cls(grid=np.linspace(s_min, s_max, n_points))

    @property
    def n_points(self) -> int:
        return len(self.grid)

    def __repr__(self) -> str:
        return (
            f"FunctionalDomain(n_points={self.n_points}, "
            f"range=[{self.grid[0]:.2f}, {self.grid[-1]:.2f}])"
        )


# ============================================================
# KERNEL E INTEGRACIÓN
# ============================================================

def gaussian_integral_kernel(
    s: np.ndarray,
    u: np.ndarray,
    bandwidth: float = 0.05,
    decay: float = 0.6,
) -> np.ndarray:
    """
    ψ(s, u) = decay · N(s; u, bandwidth²)

    Con s:(T,1) y u:(1,T) devuelve la matriz (T,T).
    """
    return (
        decay
        * np.exp(-((s - u) ** 2) / (2 * bandwidth ** 2))
        / np.sqrt(2 * np.pi * bandwidth ** 2)
    )


def build_integral_matrix(
    domain: FunctionalDomain,
    kernel_fn: Callable,
    **kernel_kwargs,
) -> np.ndarray:
    """
    Construye Ψ (T×T) como aproximación cuadratura rectangular del operador:

        (Ψ f)(s) = ∫₀¹ ψ(s,u) f(u) du  ≈  Σ_j ψ(s_i, u_j) f(u_j) Δu

    Nota: el kernel gaussiano se trunca en los bordes del dominio; las filas
    correspondientes a s≈0 y s≈1 tienen integrales ligeramente menores a `decay`.
    Esto es esperado y no constituye un error de implementación.

    Returns
    -------
    Psi : ndarray (n_points, n_points)
    """
    s = domain.grid.reshape(-1, 1)
    u = domain.grid.reshape(1, -1)
    Psi = kernel_fn(s, u, **kernel_kwargs)
    ds = np.mean(np.diff(domain.grid))
    Psi *= ds
    return Psi


# ============================================================
# SIMULADOR FAR(1)
# ============================================================

class FARSimulator:
    """
    Simulador funcional FAR(1) con tendencia determinística.

    Modelo
    ------
        X_t(s) = μ_t(s) + Y_t(s)
        Y_t(s) = (Ψ Y_{t-1})(s) + ε_t(s)

    donde ε_t ~ FBM suave (gaussian_filter1d) renormalizado a std = noise_std.

    Parámetros
    ----------
    domain : FunctionalDomain
        Malla de evaluación funcional.
    n_curves : int
        Número de curvas a generar (T en la notación de la literatura).
    Psi : np.ndarray
        Matriz del operador integral (n_points × n_points).
        Debe ser cuadrada y compatible con domain.n_points.
    trend : str o Callable
        Tendencia determinística μ_t(s).
        Claves disponibles: 'zero', 'linear', 'sinusoidal', 'bump'.
        Si es Callable: firma (t: int, s: np.ndarray) -> np.ndarray.
    noise_std : float
        Desviación estándar efectiva del ruido funcional ε_t.
        [CORRECCIÓN] Para noise_type='smooth', el vector suavizado se
        renormaliza para que std(ε_t) = noise_std exactamente.
    noise_type : str
        'white' — ruido gaussiano sin correlación espacial.
        'smooth' — ruido suavizado con gaussian_filter1d(sigma=2).
    burn_in : int
        Iteraciones de calentamiento para estabilizar la dinámica FAR(1).
        Valor por defecto: 100 (antes era 50).
    random_state : int, optional
        Semilla para reproducibilidad.
    """

    # [CORRECCIÓN 2] Tendencias normalizadas para que la amplitud sea
    # independiente de n_curves: se usan coeficientes fijos en t ∈ [0,1]
    # escalando por t/n_curves en lugar de t crudo.
    TRENDS: dict[str, Callable] = {
        "zero":
            lambda t, s, n: np.zeros_like(s),
        "linear":
            lambda t, s, n: 2.0 * (t / max(n - 1, 1)) * s,
        "sinusoidal":
            lambda t, s, n: np.sin(2 * np.pi * s) + 0.5 * np.sin(2 * np.pi * t / max(n - 1, 1)),
        "bump":
            lambda t, s, n: (1.0 + (t / max(n - 1, 1))) * np.exp(-((s - 0.5) ** 2) / 0.02),
    }

    def __init__(
        self,
        domain: FunctionalDomain,
        n_curves: int,
        Psi: np.ndarray,
        trend: Union[str, Callable] = "zero",
        noise_std: float = 1.0,
        noise_type: str = "smooth",
        burn_in: int = 100,
        random_state: Optional[int] = None,
    ):
        # [CORRECCIÓN 3] Validaciones al construir
        if not isinstance(domain, FunctionalDomain):
            raise TypeError("domain debe ser FunctionalDomain.")
        if Psi.ndim != 2 or Psi.shape[0] != Psi.shape[1]:
            raise ValueError(f"Psi debe ser cuadrada; se recibió shape {Psi.shape}.")
        if Psi.shape[0] != domain.n_points:
            raise ValueError(
                f"Psi.shape[0]={Psi.shape[0]} no coincide con domain.n_points={domain.n_points}."
            )
        if noise_std <= 0:
            raise ValueError("noise_std debe ser > 0.")
        if noise_type not in ("white", "smooth"):
            raise ValueError("noise_type debe ser 'white' o 'smooth'.")

        self.domain = domain
        self.n_curves = n_curves
        self.Psi = Psi
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.burn_in = burn_in
        self.rng = np.random.default_rng(random_state)

        # Resolución de la tendencia
        if isinstance(trend, str):
            if trend not in self.TRENDS:
                raise ValueError(
                    f"Tendencia '{trend}' no reconocida. "
                    f"Opciones: {list(self.TRENDS.keys())} o un callable."
                )
            _base = self.TRENDS[trend]
            # Envolver para inyectar n_curves automáticamente
            self.trend_fn: Callable = lambda t, s: _base(t, s, self.n_curves)
        else:
            # Verificar mínimamente la firma del callable
            sig = inspect.signature(trend)
            if len(sig.parameters) < 2:
                raise ValueError(
                    "El callable de tendencia debe aceptar al menos 2 argumentos: "
                    "(t: int, s: np.ndarray)."
                )
            self.trend_fn = trend

        # Advertencia de estabilidad
        max_eig = np.max(np.abs(np.linalg.eigvals(Psi)))
        if max_eig >= 1.0:
            import warnings
            warnings.warn(
                f"El operador Psi tiene |λ|_max = {max_eig:.4f} ≥ 1. "
                "El proceso FAR(1) puede ser NO estacionario.",
                UserWarning,
                stacklevel=2,
            )

        self.data_: Optional[np.ndarray] = None
        self._max_eigenvalue = max_eig

    # ---- Ruido ----

    def _draw_noise(self) -> np.ndarray:
        """
        Genera ε_t ∈ ℝ^{n_points} con std = noise_std.

        [CORRECCIÓN 1] Para noise_type='smooth':
          1. Se genera z ~ N(0, noise_std²).
          2. Se aplica gaussian_filter1d(sigma=2) — esto reduce std en ≈ 37%.
          3. Se renormaliza: z_smooth *= noise_std / std(z_smooth).
          Así std(ε_t) = noise_std exactamente.
        """
        z = self.rng.normal(loc=0.0, scale=self.noise_std, size=self.domain.n_points)
        if self.noise_type == "white":
            return z
        # smooth
        z_s = gaussian_filter1d(z, sigma=2.0)
        std_s = z_s.std()
        if std_s > 1e-12:           # protección contra vector constante
            z_s *= self.noise_std / std_s
        return z_s

    # ---- Simulación principal ----

    def simulate(self) -> np.ndarray:
        """
        Genera la matriz de datos funcionales X ∈ ℝ^{n_curves × n_points}.

        Returns
        -------
        X : ndarray (n_curves, n_points)
            Fila t corresponde a X_t(s).
        """
        X = np.zeros((self.n_curves, self.domain.n_points))

        # Inicialización Y_0
        y_prev = self._draw_noise()

        # Burn-in
        for _ in range(self.burn_in):
            eps = self._draw_noise()
            y_prev = self.Psi @ y_prev + eps

        # Simulación
        for t in range(self.n_curves):
            eps_t = self._draw_noise()
            y_t = self.Psi @ y_prev + eps_t
            mu_t = self.trend_fn(t, self.domain.grid)
            X[t, :] = mu_t + y_t
            y_prev = y_t

        self.data_ = X
        return X

    # ---- Diagnóstico ----

    def summary_stats(self, n_lags: int = 10) -> dict:
        """
        [NUEVO] Retorna métricas de validación del proceso simulado.

        Requiere haber llamado a .simulate() primero.
        """
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        X = self.data_
        mean_fn = X.mean(axis=0)
        std_fn = X.std(axis=0)

        # ACF en s = 0.5
        idx = np.argmin(np.abs(self.domain.grid - 0.5))
        series = X[:, idx]
        acf = [float(np.corrcoef(series[:-lag], series[lag:])[0, 1]) for lag in range(1, n_lags + 1)]

        # Test de normalidad en s = 0.5
        _, p_norm = normaltest(series)

        # Eigenvalor máximo de covarianza empírica
        Cov = np.cov(X.T)
        max_eig_cov = float(np.linalg.eigvalsh(Cov).max())

        return {
            "mean_fn_max_abs": float(np.abs(mean_fn).max()),
            "std_fn_mean": float(std_fn.mean()),
            "acf_lag1": acf[0],
            "acf": acf,
            "normality_p_at_s05": float(p_norm),
            "max_eigenvalue_operator": float(self._max_eigenvalue),
            "max_eigenvalue_cov_empirical": max_eig_cov,
        }

    # ---- Visualización ----

    def plot_curves(
        self,
        n_show: int = 50,
        alpha: float = 0.3,
        title: Optional[str] = None,
        figsize=(10, 5),
    ) -> None:
        """Curvas funcionales generadas X_t(s)."""
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        fig, ax = plt.subplots(figsize=figsize)
        for t in range(min(n_show, self.n_curves)):
            ax.plot(self.domain.grid, self.data_[t], alpha=alpha, color="steelblue", lw=0.8)
        # Media funcional en overlay
        ax.plot(
            self.domain.grid, self.data_.mean(axis=0),
            color="crimson", lw=2.0, label="Media empírica",
        )
        ax.set_xlabel("s (dominio funcional)")
        ax.set_ylabel("X_t(s)")
        ax.set_title(title or f"FAR(1) — {self.n_curves} curvas")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_heatmap(
        self,
        title: Optional[str] = None,
        figsize=(10, 6),
    ) -> None:
        """Heatmap X_t(s) con escala robusta (percentiles 1–99)."""
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            self.data_,
            aspect="auto",
            cmap="RdBu_r",
            extent=[
                self.domain.grid.min(),
                self.domain.grid.max(),
                self.n_curves,
                0,
            ],
            vmin=np.percentile(self.data_, 1),
            vmax=np.percentile(self.data_, 99),
        )
        ax.set_xlabel("s")
        ax.set_ylabel("t (índice de curva)")
        ax.set_title(title or "Heatmap X_t(s) — FAR(1)")
        plt.colorbar(im, ax=ax, label="X_t(s)")
        plt.tight_layout()
        plt.show()

    def plot_acf(
        self,
        s_points: Optional[list] = None,
        n_lags: int = 20,
        title: Optional[str] = None,
        figsize=(10, 4),
    ) -> None:
        """
        [NUEVO] Autocorrelación temporal de X_t(s) en puntos s fijos.

        Útil para verificar visualmente la dependencia FAR(1).
        """
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        s_points = s_points or [0.2, 0.5, 0.8]
        fig, ax = plt.subplots(figsize=figsize)
        for s_val in s_points:
            idx = np.argmin(np.abs(self.domain.grid - s_val))
            series = self.data_[:, idx]
            acf = [1.0] + [
                float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
                for lag in range(1, n_lags + 1)
            ]
            ax.plot(range(len(acf)), acf, marker="o", markersize=3, label=f"s={s_val}")
        # Banda de significancia ≈ 2/√T
        ci = 2 / np.sqrt(self.n_curves)
        ax.axhline(ci, color="gray", ls="--", lw=0.8, label=f"±2/√T = ±{ci:.2f}")
        ax.axhline(-ci, color="gray", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(title or "Autocorrelación funcional (puntos s fijos)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"domain={self.domain.n_points} pts, "
            f"n_curves={self.n_curves}, "
            f"burn_in={self.burn_in}, "
            f"|λ|_max={self._max_eigenvalue:.4f})"
        )


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    domain = FunctionalDomain.regular(s_min=0, s_max=1, n_points=100)

    Psi = build_integral_matrix(
        domain,
        gaussian_integral_kernel,
        bandwidth=0.05,
        decay=0.6,
    )

    sim = FARSimulator(
        domain=domain,
        n_curves=200,
        Psi=Psi,
        trend="sinusoidal",
        noise_std=0.5,
        noise_type="smooth",
        burn_in=100,
        random_state=42,
    )

    X = sim.simulate()
    print(sim)
    print(f"Datos generados: {X.shape}")

    stats = sim.summary_stats()
    print(f"\n--- Diagnóstico ---")
    print(f"  |media funcional|_max : {stats['mean_fn_max_abs']:.4f}")
    print(f"  std funcional medio   : {stats['std_fn_mean']:.4f}")
    print(f"  ACF lag-1 en s=0.5   : {stats['acf_lag1']:.4f}")
    print(f"  p-value normalidad   : {stats['normality_p_at_s05']:.4f}")

    sim.plot_curves(n_show=40, title="FAR(1) con tendencia sinusoidal")
    sim.plot_heatmap(title="Heatmap FAR(1)")
    sim.plot_acf(title="ACF temporal por punto s")
