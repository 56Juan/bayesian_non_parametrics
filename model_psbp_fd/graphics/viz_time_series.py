"""
viz_time_series.py
==================
Visualizadores de series de tiempo funcionales en formato continuo desplazado.

Cada curva X_t(s) ocupa el intervalo [t-1, t] del eje horizontal, de modo
que la figura se lee igual que una serie escalar: de izquierda a derecha
en el tiempo. Las líneas verticales grises separan observaciones funcionales.

Funciones públicas
------------------
plot_fts_empirical      : Serie de tiempo con curvas empíricas discretas.
plot_fts_functional     : Serie de tiempo con curvas reconstruidas mediante
                          una representación funcional (fr.reconstruct).
plot_fts_comparison     : Comparativa completa en todo T: empírica | repr.
                          funcional | predicción PSBP. Las tres capas
                          superpuestas, con leyenda y métricas por instante.

Convención de datos
-------------------
X         : (T, G)  — curvas (empíricas o estandarizadas).
grid      : (G,)    — puntos del dominio s ∈ [0, 1] (o cualquier intervalo).
fr        : instancia de FunctionalRepresentation ya ajustada (post fit).
X_pred_repr : (T, G) — curvas reconstruidas a partir de scores predichos
              por el modelo PSBP (fr.reconstruct(THETA_pred)).

Notas de diseño
---------------
- El desplazamiento temporal usa la convención X_t en [t-1, t], idéntica
  a la función original `plot_functional_time_series`.
- Las tres funciones comparten el mismo helper interno `_build_continuous`
  para garantizar coherencia en el desplazamiento.
- `plot_fts_comparison` acepta T completo y N_LAGS para alinear
  correctamente X_pred_repr (que solo existe para t = N_LAGS+1 … T).
  Los primeros N_LAGS instantes se muestran solo con empírica y repr.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Helper interno compartido
# ─────────────────────────────────────────────────────────────────────────────

def _build_continuous(
    X: np.ndarray,
    grid: np.ndarray,
    t_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye los arrays (x_global, y_global) para una serie continua
    desplazada en el eje temporal.

    Cada fila X[t] se mapea al intervalo [t_offset + t, t_offset + t + 1].

    Parámetros
    ----------
    X        : (T, G)
    grid     : (G,) en [0, 1] normalizado; si no lo está se normaliza internamente.
    t_offset : desplazamiento base (útil para alinear sub-series en plot_comparison).

    Retorna
    -------
    x_global : (T*G,)
    y_global : (T*G,)
    """
    T, G = X.shape
    # Normalizar grid a [0, 1] para el desplazamiento
    g_min, g_max = grid.min(), grid.max()
    grid_norm = (grid - g_min) / (g_max - g_min) if g_max > g_min else grid

    x_global = np.concatenate([
        (t_offset + t) + grid_norm for t in range(T)
    ])
    y_global = X.ravel()
    return x_global, y_global


def _add_separators(
    ax: plt.Axes,
    T: int,
    t_offset: int = 0,
    separator_every: int = 1,
) -> None:
    """Dibuja líneas verticales separadoras entre observaciones."""
    for t in range(1, T, separator_every):
        ax.axvline(t_offset + t, color="#cccccc", lw=0.6, zorder=0)


def _add_labels(
    ax: plt.Axes,
    X: np.ndarray,
    highlight_idx: List[int],
    t_offset: int = 0,
    color: str = "#1a6faf",
    y_range: Optional[float] = None,
) -> None:
    """Añade anotaciones sobre las curvas destacadas."""
    T = X.shape[0]
    y_range = y_range or (X.max() - X.min())
    for t in highlight_idx:
        x_mid = t_offset + t + 0.5
        y_top = X[t].max()
        label = (
            r"$X_1(s)$"        if t == 0
            else r"$X_T(s)$"   if t == T - 1
            else fr"$X_{{{t + 1}}}(s)$"
        )
        ax.annotate(
            label,
            xy=(x_mid, y_top),
            xytext=(x_mid, y_top + 0.18 * y_range),
            ha="center", fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7),
        )


def _format_time_axis(
    ax: plt.Axes,
    T: int,
    t_offset: int = 0,
) -> None:
    """Configura ticks y límites del eje temporal."""
    tick_step = max(1, T // 10)
    ticks = list(range(t_offset, t_offset + T + 1, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v - t_offset) for v in ticks], fontsize=8)
    ax.set_xlim(t_offset, t_offset + T)


# ─────────────────────────────────────────────────────────────────────────────
# Función 1 — Serie empírica
# ─────────────────────────────────────────────────────────────────────────────

def plot_fts_empirical(
    X: np.ndarray,
    grid: np.ndarray,
    highlight_idx: Optional[List[int]] = None,
    color: str = "#1a6faf",
    title: Optional[str] = None,
    separator_every: int = 1,
    figsize: Tuple[float, float] = (14, 4),
    dpi: int = 130,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Serie de tiempo funcional con curvas empíricas discretas.

    Cada X_t(s) ocupa el intervalo [t-1, t] del eje horizontal.
    Las curvas se grafican tal cual provienen de los datos (sin reconstrucción).

    Parámetros
    ----------
    X              : (T, G) curvas empíricas
    grid           : (G,) dominio s
    highlight_idx  : índices t (0-based) a etiquetar; default [0, 1, T//2, T-1]
    color          : color de la línea continua
    title          : título (auto si None)
    separator_every: cada cuántos t dibujar separador vertical
    figsize        : tamaño de la figura
    dpi            : resolución
    save_path      : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    T, G = X.shape
    highlight_idx = highlight_idx or [0, 1, T // 2, T - 1]

    x_g, y_g = _build_continuous(X, grid)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x_g, y_g, color=color, lw=0.85, alpha=0.92)

    _add_separators(ax, T, separator_every=separator_every)
    _add_labels(ax, X, highlight_idx, color=color)
    _format_time_axis(ax, T)

    ax.set_xlabel(
        r"Tiempo $t$  (intervalo $[t-1,\,t]$ = curva $X_t(s)$)", fontsize=9
    )
    ax.set_ylabel(r"$X_t(s)$", fontsize=10)
    ax.set_title(
        title or f"Serie de tiempo funcional — empírica  ($T={T}$)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Función 2 — Serie con representación funcional
# ─────────────────────────────────────────────────────────────────────────────

def plot_fts_functional(
    X: np.ndarray,
    grid: np.ndarray,
    fr,
    highlight_idx: Optional[List[int]] = None,
    color: str = "#e07b39",
    title: Optional[str] = None,
    separator_every: int = 1,
    figsize: Tuple[float, float] = (14, 4),
    dpi: int = 130,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Serie de tiempo funcional con curvas reconstruidas mediante la
    representación funcional (B-spline, Fourier, FPCA, etc.).

    Reconstruye X_hat = fr.reconstruct(fr.transform(X)) y aplica
    el mismo desplazamiento temporal que plot_fts_empirical.

    Parámetros
    ----------
    X              : (T, G) curvas empíricas (input para fr.transform)
    grid           : (G,) dominio s
    fr             : instancia FunctionalRepresentation ya ajustada
    highlight_idx  : índices t (0-based) a etiquetar
    color          : color de la línea continua
    title          : título (auto si None)
    separator_every: cada cuántos t dibujar separador vertical
    figsize        : tamaño de la figura
    dpi            : resolución
    save_path      : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    T, G = X.shape
    highlight_idx = highlight_idx or [0, 1, T // 2, T - 1]

    # Reconstrucción funcional
    THETA  = fr.transform(X, grid)
    X_repr = fr.reconstruct(THETA)          # (T, G)

    x_g, y_g = _build_continuous(X_repr, grid)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x_g, y_g, color=color, lw=0.85, alpha=0.92)

    _add_separators(ax, T, separator_every=separator_every)
    _add_labels(ax, X_repr, highlight_idx, color=color)
    _format_time_axis(ax, T)

    method_label = getattr(fr, "method", "funcional")
    ax.set_xlabel(
        r"Tiempo $t$  (intervalo $[t-1,\,t]$ = curva $\hat{X}_t(s)$)", fontsize=9
    )
    ax.set_ylabel(r"$\hat{X}_t(s)$", fontsize=10)
    ax.set_title(
        title or (
            f"Serie de tiempo funcional — representación {method_label}  ($T={T}$)"
        ),
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Función 3 — Comparativa completa en todo T
# ─────────────────────────────────────────────────────────────────────────────

def plot_fts_comparison(
    X: np.ndarray,
    grid: np.ndarray,
    fr,
    X_pred_repr: np.ndarray,
    n_lags: int = 1,
    highlight_idx: Optional[List[int]] = None,
    separator_every: int = 1,
    color_empirical:  str = "#2c3e50",
    color_functional: str = "#1a6faf",
    color_predicted:  str = "#c0392b",
    alpha_empirical:  float = 0.55,
    alpha_functional: float = 0.80,
    alpha_predicted:  float = 0.92,
    lw_empirical:     float = 0.70,
    lw_functional:    float = 1.00,
    lw_predicted:     float = 1.20,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 5),
    dpi: int = 130,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, dict]:
    """
    Comparativa completa en todo T de tres capas superpuestas:

        (gris oscuro) X_t empírica              — datos originales
        (azul)        X̂_t repr. funcional       — fr.reconstruct(fr.transform(X))
        (rojo)        X̂_t predicción PSBP       — fr.reconstruct(THETA_pred)

    Las primeras N_LAGS observaciones no tienen predicción PSBP (el modelo
    necesita lags), por lo que solo se grafican las dos primeras capas en
    ese tramo. Una franja vertical sombreada marca el período sin predicción.

    Parámetros
    ----------
    X              : (T, G) curvas empíricas completas (toda la serie)
    grid           : (G,) dominio s
    fr             : instancia FunctionalRepresentation ya ajustada
    X_pred_repr    : (T_eff, G) curvas predichas reconstruidas, donde
                     T_eff = T - n_lags. Obtenidas como:
                         fr.reconstruct(THETA_pred)
                     con THETA_pred = columna "y_hat" de eval_results por k.
    n_lags         : número de lags del modelo AR(p); determina el offset
                     entre X y X_pred_repr en el eje temporal
    highlight_idx  : índices t (0-based en T completo) a etiquetar
    separator_every: cada cuántos t dibujar separador vertical
    color_*        : colores para cada capa
    alpha_*        : transparencias para cada capa
    lw_*           : grosores de línea para cada capa
    title          : título (auto si None)
    figsize        : tamaño de la figura
    dpi            : resolución
    save_path      : ruta de guardado opcional

    Retorna
    -------
    fig     : Figure
    metrics : dict con RMSE_repr, RMSE_pred y RMSE_total
              (calculados solo sobre T_eff, escala estandarizada)

    Notas
    -----
    X_pred_repr se construye fuera de este módulo:
        THETA_pred  = np.column_stack([eval_results[k]["y_hat"] for k in range(K)])
        X_pred_repr = fr.reconstruct(THETA_pred)
    """
    T, G    = X.shape
    T_eff   = T - n_lags
    assert X_pred_repr.shape == (T_eff, G), (
        f"X_pred_repr debe ser ({T_eff}, {G}); recibido {X_pred_repr.shape}. "
        f"Verifica que T_eff = T - n_lags = {T} - {n_lags} = {T_eff}."
    )

    highlight_idx = highlight_idx or [0, 1, T // 2, T - 1]

    # ── Reconstrucción funcional completa (todo T) ────────────────────────
    THETA_all = fr.transform(X, grid)
    X_repr    = fr.reconstruct(THETA_all)               # (T, G)

    # ── Métricas (solo sobre T_eff para comparabilidad con predicción) ───
    X_true_eff = X[n_lags:, :]                          # (T_eff, G)
    X_repr_eff = X_repr[n_lags:, :]                     # (T_eff, G)

    rmse_repr = float(np.sqrt(np.mean((X_true_eff - X_repr_eff)    ** 2)))
    rmse_pred = float(np.sqrt(np.mean((X_true_eff - X_pred_repr)   ** 2)))
    rmse_tot  = float(np.sqrt(np.mean((X_true_eff - X_pred_repr)   ** 2)))
    sd        = float(X_true_eff.std())
    metrics   = {
        "rmse_repr":   rmse_repr,
        "rmse_pred":   rmse_pred,
        "nrmse_repr":  rmse_repr / sd if sd > 0 else np.nan,
        "nrmse_pred":  rmse_pred / sd if sd > 0 else np.nan,
    }

    # ── Construir arrays continuos ────────────────────────────────────────
    # Capa 1: empírica — todo T, offset=0
    x_emp, y_emp = _build_continuous(X,       grid, t_offset=0)
    # Capa 2: repr funcional — todo T, offset=0
    x_rep, y_rep = _build_continuous(X_repr,  grid, t_offset=0)
    # Capa 3: predicción — solo T_eff, offset=n_lags
    x_prd, y_prd = _build_continuous(X_pred_repr, grid, t_offset=n_lags)

    # ── Figura ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Franja sin predicción (primeros n_lags instantes)
    if n_lags > 0:
        ax.axvspan(0, n_lags, color="#f0f0f0", zorder=0,
                   label=f"sin predicción (N_LAGS={n_lags})")
        ax.axvline(n_lags, color="#999999", lw=1.0, ls="--", zorder=1)

    # Tres capas superpuestas
    ax.plot(x_emp, y_emp,
            color=color_empirical,  lw=lw_empirical,  alpha=alpha_empirical,
            label="Empírica")
    ax.plot(x_rep, y_rep,
            color=color_functional, lw=lw_functional, alpha=alpha_functional,
            label="Repr. funcional")
    ax.plot(x_prd, y_prd,
            color=color_predicted,  lw=lw_predicted,  alpha=alpha_predicted,
            label="Predicción PSBP")

    # Separadores
    _add_separators(ax, T, separator_every=separator_every)

    # Etiquetas sobre curvas destacadas (usando la empírica como referencia
    # de posición y rango)
    y_range = X.max() - X.min()
    _add_labels(ax, X, highlight_idx, color=color_empirical, y_range=y_range)

    # Anotación de métricas en la esquina
    metrics_txt = (
        f"RMSE repr = {rmse_repr:.4f}  (NRMSE={metrics['nrmse_repr']:.3f})\n"
        f"RMSE pred = {rmse_pred:.4f}  (NRMSE={metrics['nrmse_pred']:.3f})"
    )
    ax.text(
        0.01, 0.97, metrics_txt,
        transform=ax.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85),
    )

    _format_time_axis(ax, T)
    ax.set_xlabel(
        r"Tiempo $t$  (intervalo $[t-1,\,t]$ = curva $X_t(s)$)", fontsize=9
    )
    ax.set_ylabel(r"$X_t(s)$", fontsize=10)

    method_label = getattr(fr, "method", "funcional")
    ax.set_title(
        title or (
            f"Comparativa serie de tiempo funcional — "
            f"empírica | repr. {method_label} | predicción PSBP  ($T={T}$)"
        ),
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, metrics
