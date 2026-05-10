import numpy as np
from typing import Callable, Optional, Union
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


@dataclass
class FunctionalDomain:
    """Malla de puntos s donde se evalúa cada curva funcional."""
    grid: np.ndarray

    @classmethod
    def regular(cls, s_min: float = 0.0, s_max: float = 1.0, n_points: int = 100):
        return cls(grid=np.linspace(s_min, s_max, n_points))

    @property
    def n_points(self) -> int:
        return len(self.grid)


def gaussian_integral_kernel(
    s: np.ndarray, u: np.ndarray, bandwidth: float = 0.05, decay: float = 0.6
) -> np.ndarray:
    """
    psi(s, u) = decay * exp(-(s-u)^2 / (2*bandwidth^2)) / sqrt(2*pi*bandwidth^2)
    s: (T, 1)   u: (1, T)   devuelve: (T, T)
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
    Construye la matriz Psi (T x T) que aproxima el operador integral
    int psi(s,u) f(u) du via cuadratura.
    """
    s = domain.grid.reshape(-1, 1)
    u = domain.grid.reshape(1, -1)
    Psi = kernel_fn(s, u, **kernel_kwargs)
    ds = np.mean(np.diff(domain.grid))
    Psi *= ds
    return Psi


class FARSimulator:
    """
    Simulador funcional FAR(1) con tendencia.

    X_t(s) = mu_t(s) + Y_t(s)
    Y_t(s) = Psi @ Y_{t-1} + epsilon_t(s)

    Parámetros
    ----------
    domain : FunctionalDomain
        Malla de evaluación funcional.
    n_curves : int
        Número de curvas a generar.
    Psi : np.ndarray
        Matriz del operador integral (n_points x n_points).
    trend : str o Callable
        Tendencia determinística. Si es str, debe ser una clave de TRENDS.
        Si es Callable, debe tener firma (t: int, s: np.ndarray) -> np.ndarray.
    noise_std : float
        Desviación estándar del ruido funcional.
    noise_type : str
        "white" o "smooth".
    burn_in : int
        Número de iteraciones de calentamiento para estabilizar la dinámica.
    random_state : int, optional
        Semilla para reproducibilidad.
    """

    TRENDS = {
        "zero": lambda t, s: np.zeros_like(s),
        "linear": lambda t, s: 0.02 * t * s,
        "sinusoidal": lambda t, s: np.sin(2 * np.pi * s) + 0.03 * t,
        "bump": lambda t, s: (1 + 0.015 * t) * np.exp(-((s - 0.5) ** 2) / 0.02),
    }

    def __init__(
        self,
        domain: FunctionalDomain,
        n_curves: int,
        Psi: np.ndarray,
        trend: Union[str, Callable[[int, np.ndarray], np.ndarray]] = "zero",
        noise_std: float = 1.0,
        noise_type: str = "smooth",
        burn_in: int = 50,
        random_state: Optional[int] = None,
    ):
        self.domain = domain
        self.n_curves = n_curves
        self.Psi = Psi
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.burn_in = burn_in
        self.rng = np.random.default_rng(random_state)

        if isinstance(trend, str):
            if trend not in self.TRENDS:
                raise ValueError(
                    f"Tendencia '{trend}' no reconocida. "
                    f"Usa: {list(self.TRENDS.keys())} o un callable."
                )
            self.trend_fn = self.TRENDS[trend]
        else:
            self.trend_fn = trend

        self.data_: Optional[np.ndarray] = None

    def _draw_noise(self) -> np.ndarray:
        """Genera un vector de ruido funcional."""
        z = self.rng.normal(loc=0.0, scale=self.noise_std, size=self.domain.n_points)
        if self.noise_type == "white":
            return z
        if self.noise_type == "smooth":
            return gaussian_filter1d(z, sigma=2.0)
        raise ValueError("noise_type debe ser 'white' o 'smooth'.")

    def simulate(self) -> np.ndarray:
        """
        Genera la matriz de datos funcionales FAR(1) con tendencia.

        Returns
        -------
        X : ndarray, shape (n_curves, n_points)
            Fila t = curva X_t(s).
        """
        X = np.zeros((self.n_curves, self.domain.n_points))
        
        # Inicialización de Y_0 con ruido
        y_prev = self._draw_noise()
        
        # Burn-in para estabilizar la dinámica del proceso FAR(1)
        for _ in range(self.burn_in):
            eps = self._draw_noise()
            y_prev = self.Psi @ y_prev + eps
        
        # Simulación principal
        for t in range(self.n_curves):
            eps_t = self._draw_noise()
            y_t = self.Psi @ y_prev + eps_t
            mu_t = self.trend_fn(t, self.domain.grid)
            X[t, :] = mu_t + y_t
            y_prev = y_t

        self.data_ = X
        return X

    def plot_curves(
        self,
        n_show: int = 50,
        alpha: float = 0.3,
        title: Optional[str] = None,
        figsize=(10, 5),
    ):
        """Visualiza las curvas funcionales generadas."""
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        
        plt.figure(figsize=figsize)
        for t in range(min(n_show, self.n_curves)):
            plt.plot(self.domain.grid, self.data_[t], alpha=alpha, color="steelblue")
        plt.xlabel("s (dominio funcional)")
        plt.ylabel("X_t(s)")
        plt.title(title or "FAR(1) funcional con tendencia")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_heatmap(
        self,
        title: Optional[str] = None,
        figsize=(10, 6),
    ):
        """Visualiza un heatmap de todas las curvas generadas."""
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
        ax.set_ylabel("t (tiempo / índice de curva)")
        ax.set_title(title or "Heatmap X_t(s) — FAR(1)")
        plt.colorbar(im, ax=ax, label="X_t(s)")
        plt.show()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"domain={self.domain.n_points} pts, "
            f"n_curves={self.n_curves}, "
            f"burn_in={self.burn_in})"
        )


# ========================================================================
# EJEMPLO DE USO
# ========================================================================

if __name__ == "__main__":
    # Crear dominio funcional
    domain = FunctionalDomain.regular(s_min=0, s_max=1, n_points=100)
    
    # Construir matriz del operador integral
    Psi = build_integral_matrix(
        domain, 
        gaussian_integral_kernel, 
        bandwidth=0.05, 
        decay=0.6
    )
    
    # Crear simulador FAR(1) con tendencia sinusoidal
    sim = FARSimulator(
        domain=domain,
        n_curves=200,
        Psi=Psi,
        trend="sinusoidal",
        noise_std=0.5,
        noise_type="smooth",
        burn_in=50,
        random_state=42
    )
    
    # Generar datos
    X = sim.simulate()
    print(f"Datos generados: {X.shape}")
    
    # Visualizar
    sim.plot_curves(n_show=30, title="FAR(1) con tendencia sinusoidal")
    sim.plot_heatmap(title="Heatmap del proceso FAR(1)")