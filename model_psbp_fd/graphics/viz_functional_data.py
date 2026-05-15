"""
viz_functional_data.py
======================
Visualizadores de datos funcionales para el pipeline PSBP-FD.

Funciones públicas
------------------
plot_functional_series     : Serie temporal completa de curvas (formato tipo
                             plot_functional_time_series, usando la representación
                             funcional reconstruida continua).
plot_empirical_sample      : Muestra aleatoria o especificada de curvas empíricas
                             en el dominio S.
plot_functional_mean       : Media funcional con banda ±1 std (y opcionalmente
                             ±2 std).
plot_functional_variance   : Varianza funcional puntual a lo largo del dominio.
plot_mean_and_variance     : Panel combinado media + varianza en una sola figura.

Contexto
--------
Los datos son curvas X(s) en un dominio S, observadas a lo largo de T.
X : (T, G)  — T curvas discretas, G puntos de grilla.
grid : (G,) — puntos del dominio S.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────────────────
# Paletas y estilos
# ─────────────────────────────────────────────────────────────────────────────
_HIGHLIGHT_COLORS = ["#c0392b", "#1a6faf", "#27ae60", "#8e44ad",
                     "#e67e22", "#2c3e50", "#16a085", "#d35400"]


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def plot_functional_series(
    X: np.ndarray,
    grid: np.ndarray,
    fr=None,
    highlight_idx: Optional[List[int]] = None,
    title: str = "Series de tiempo funcionales",
    xlabel: str = "s",
    ylabel: str = r"$X_t(s)$",
    alpha_bg: float = 0.18,
    color_bg: str = "steelblue",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Muestra la serie de tiempo funcional completa.

    Si se pasa `fr` (instancia de FunctionalRepresentation ajustada), muestra
    las curvas reconstruidas (representación funcional continua). Si no,
    muestra las curvas empíricas discretas.

    Parámetros
    ----------
    X            : (T, G) curvas (empíricas o estandarizadas)
    grid         : (G,) dominio S
    fr           : FunctionalRepresentation ajustada (opcional). Si se pasa,
                   se muestra X_hat = fr.reconstruct(fr.transform(X)).
    highlight_idx: índices t a resaltar con colores distintos (máx 8)
    title        : título de la figura
    xlabel       : etiqueta eje x
    ylabel       : etiqueta eje y
    alpha_bg     : transparencia de las curvas de fondo
    color_bg     : color de las curvas de fondo
    figsize      : tamaño de la figura
    save_path    : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    T, G = X.shape
    if fr is not None:
        # Representación funcional reconstruida (continua)
        THETA = fr.transform(X, grid)
        X_plot = fr.reconstruct(THETA)
        label_source = "Representación funcional"
    else:
        X_plot = X
        label_source = "Curvas empíricas"

    fig, ax = plt.subplots(figsize=figsize)

    # ── Curvas de fondo ──────────────────────────────────────────────────
    hi_set = set(highlight_idx or [])
    for t in range(T):
        if t not in hi_set:
            ax.plot(grid, X_plot[t], color=color_bg, lw=0.5, alpha=alpha_bg)

    # ── Curvas destacadas ────────────────────────────────────────────────
    if highlight_idx:
        for i, t in enumerate(highlight_idx[:len(_HIGHLIGHT_COLORS)]):
            ax.plot(grid, X_plot[t],
                    color=_HIGHLIGHT_COLORS[i], lw=1.8,
                    label=f"t = {t}", zorder=3)
        ax.legend(fontsize=8, loc="upper right")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}\n({label_source}, T={T})", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_empirical_sample(
    X: np.ndarray,
    grid: np.ndarray,
    sample_idx: Optional[List[int]] = None,
    n_sample: int = 5,
    seed: Optional[int] = None,
    title: str = "Muestra de curvas empíricas",
    xlabel: str = "s",
    ylabel: str = r"$X_t(s)$",
    figsize: Tuple[float, float] = (9, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grafica una muestra de curvas empíricas en el dominio S.

    Si se especifica `sample_idx`, usa esos índices. Si no, toma
    `n_sample` índices aleatorios controlados por `seed`.

    Parámetros
    ----------
    X          : (T, G) curvas empíricas
    grid       : (G,) dominio S
    sample_idx : lista de índices t a graficar (sobreescribe n_sample y seed)
    n_sample   : número de curvas a muestrear si sample_idx es None
    seed       : semilla para la selección aleatoria
    title      : título de la figura
    xlabel     : etiqueta eje x
    ylabel     : etiqueta eje y
    figsize    : tamaño de la figura
    save_path  : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    T = X.shape[0]

    if sample_idx is None:
        rng = np.random.default_rng(seed)
        sample_idx = sorted(rng.choice(T, size=min(n_sample, T), replace=False).tolist())

    n_show = len(sample_idx)
    colors = _HIGHLIGHT_COLORS[:n_show] if n_show <= len(_HIGHLIGHT_COLORS) else (
        [cm.tab10(i / n_show) for i in range(n_show)]
    )

    fig, ax = plt.subplots(figsize=figsize)

    for i, t in enumerate(sample_idx):
        ax.plot(grid, X[t], color=colors[i], lw=1.6, label=f"t = {t}")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}  (n={n_show})", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_functional_mean(
    X: np.ndarray,
    grid: np.ndarray,
    show_std1: bool = True,
    show_std2: bool = False,
    color: str = "darkblue",
    title: str = "Media funcional",
    xlabel: str = "s",
    ylabel: str = r"$\bar{X}(s)$",
    figsize: Tuple[float, float] = (9, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Media funcional muestral con bandas de confianza.

    Parámetros
    ----------
    X          : (T, G) curvas
    grid       : (G,) dominio S
    show_std1  : muestra banda ±1 std (True por defecto)
    show_std2  : muestra banda ±2 std (False por defecto)
    color      : color de la línea media
    title      : título
    xlabel     : etiqueta eje x
    ylabel     : etiqueta eje y
    figsize    : tamaño de la figura
    save_path  : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    mean_curve = X.mean(axis=0)
    std_curve  = X.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=figsize)

    if show_std2:
        ax.fill_between(grid,
                        mean_curve - 2 * std_curve,
                        mean_curve + 2 * std_curve,
                        color=color, alpha=0.12, label="±2 std")
    if show_std1:
        ax.fill_between(grid,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=color, alpha=0.25, label="±1 std")

    ax.plot(grid, mean_curve, color=color, lw=2.5, label="Media funcional")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}  (T={X.shape[0]})", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_functional_variance(
    X: np.ndarray,
    grid: np.ndarray,
    color: str = "crimson",
    title: str = "Varianza funcional",
    xlabel: str = "s",
    ylabel: str = r"$\mathrm{Var}\,X(s)$",
    figsize: Tuple[float, float] = (9, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Varianza funcional puntual a lo largo del dominio S.

    Parámetros
    ----------
    X         : (T, G) curvas
    grid      : (G,) dominio S
    color     : color de la curva
    title     : título
    xlabel    : etiqueta eje x
    ylabel    : etiqueta eje y
    figsize   : tamaño de la figura
    save_path : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    var_curve = X.var(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(grid, 0, var_curve, color=color, alpha=0.2)
    ax.plot(grid, var_curve, color=color, lw=2.0, label="Varianza funcional")
    ax.axhline(var_curve.mean(), color="k", lw=1.0, ls="--",
               label=f"Varianza media = {var_curve.mean():.4f}")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}  (T={X.shape[0]})", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_mean_and_variance(
    X: np.ndarray,
    grid: np.ndarray,
    show_std1: bool = True,
    show_std2: bool = False,
    color_mean: str = "darkblue",
    color_var: str = "crimson",
    title: str = "Media y varianza funcional",
    xlabel: str = "s",
    figsize: Tuple[float, float] = (13, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Panel combinado: media funcional (izquierda) y varianza funcional (derecha).

    Parámetros
    ----------
    X           : (T, G) curvas
    grid        : (G,) dominio S
    show_std1   : muestra banda ±1 std en el panel de media
    show_std2   : muestra banda ±2 std en el panel de media
    color_mean  : color del panel de media
    color_var   : color del panel de varianza
    title       : título global
    xlabel      : etiqueta eje x compartido
    figsize     : tamaño total de la figura
    save_path   : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    mean_curve = X.mean(axis=0)
    std_curve  = X.std(axis=0, ddof=1)
    var_curve  = std_curve ** 2

    fig, (ax_m, ax_v) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{title}  (T={X.shape[0]})", fontsize=12)

    # ── Panel media ──────────────────────────────────────────────────────
    if show_std2:
        ax_m.fill_between(grid,
                          mean_curve - 2 * std_curve,
                          mean_curve + 2 * std_curve,
                          color=color_mean, alpha=0.12, label="±2 std")
    if show_std1:
        ax_m.fill_between(grid,
                          mean_curve - std_curve,
                          mean_curve + std_curve,
                          color=color_mean, alpha=0.25, label="±1 std")
    ax_m.plot(grid, mean_curve, color=color_mean, lw=2.5, label="Media funcional")
    ax_m.set_xlabel(xlabel, fontsize=10)
    ax_m.set_ylabel(r"$\bar{X}(s)$", fontsize=10)
    ax_m.set_title("Media funcional", fontsize=10)
    ax_m.legend(fontsize=8, loc="best")
    ax_m.grid(True, alpha=0.35)

    # ── Panel varianza ───────────────────────────────────────────────────
    ax_v.fill_between(grid, 0, var_curve, color=color_var, alpha=0.2)
    ax_v.plot(grid, var_curve, color=color_var, lw=2.0,
              label="Varianza funcional")
    ax_v.axhline(var_curve.mean(), color="k", lw=1.0, ls="--",
                 label=f"Media = {var_curve.mean():.4f}")
    ax_v.set_xlabel(xlabel, fontsize=10)
    ax_v.set_ylabel(r"$\mathrm{Var}\,X(s)$", fontsize=10)
    ax_v.set_title("Varianza funcional", fontsize=10)
    ax_v.legend(fontsize=8, loc="best")
    ax_v.grid(True, alpha=0.35)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
