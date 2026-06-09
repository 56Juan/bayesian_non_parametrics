"""
viz_preprocessing.py
====================
Visualizadores del pipeline de preprocesamiento para PSBP-FD.

Funciones públicas
------------------
plot_diagnostico_estandarizacion : Panel media/std antes y después de estandarizar.
plot_seleccion_basis              : Triple panel de selección B-spline (heatmap VR,
                                    heatmap RMSE, curva VR vs n_basis).

Contexto
--------
Estas funciones cubren el diagnóstico visual de la sección §2.3 (estandarización)
y §3 (selección de la base B-spline) del notebook 01_02_modelo.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Paleta de colores por orden B-spline
# ─────────────────────────────────────────────────────────────────────────────
_COLORS_ORDER = ["#1a6faf", "#e07b39", "#3aaa35", "#9b59b6"]


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagnostico_estandarizacion(
    X_raw: np.ndarray,
    X_std: np.ndarray,
    grid: np.ndarray,
    labels: Tuple[str, str] = ("Original", "Estandarizada"),
    color_mean: str = "steelblue",
    color_std: str = "crimson",
    title: str = "Estadísticas marginales por columna",
    figsize: Tuple[float, float] = (13, 4),
    save_path: Optional[str] = None,
    dpi: int = 130,
) -> plt.Figure:
    """
    Doble panel: media y std por punto de grilla, antes y después de estandarizar.

    Parámetros
    ----------
    X_raw      : (T, G) datos sin estandarizar
    X_std      : (T, G) datos estandarizados
    grid       : (G,) puntos del dominio S
    labels     : etiquetas para el panel izquierdo y derecho
    color_mean : color de la curva de media
    color_std  : color de la curva de std (línea discontinua)
    title      : título base de cada panel
    figsize    : tamaño de la figura
    save_path  : ruta de guardado opcional
    dpi        : resolución al guardar

    Retorna
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, Xp, lbl in zip(axes, [X_raw, X_std], labels):
        ax.plot(grid, Xp.mean(axis=0), color=color_mean, lw=1.5, label="media")
        ax.plot(grid, Xp.std(axis=0),  color=color_std,  lw=1.5, ls="--", label="std")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.set_title(f"X {lbl}: {title}", fontsize=10)
        ax.set_xlabel("s")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_seleccion_basis(
    sel_df: pd.DataFrame,
    nb_best: int,
    ord_best: int,
    figsize: Tuple[float, float] = (18, 5),
    save_path: Optional[str] = None,
    dpi: int = 130,
) -> plt.Figure:
    """
    Triple panel de selección de base B-spline.

    Paneles:
    - Izquierda  : Heatmap de varianza retenida (%) por (order, n_basis).
    - Centro     : Heatmap de RMSE medio de reconstrucción por (order, n_basis).
    - Derecha    : Curva de varianza retenida vs n_basis, una línea por order.

    El par (nb_best, ord_best) se destaca con un rectángulo rojo en los heatmaps
    y una línea vertical en la curva.

    Parámetros
    ----------
    sel_df   : DataFrame con columnas [n_basis, order, var_retained, rmse_mean].
               Típicamente producido por el grid-search de FunctionalRepresentation.
    nb_best  : n_basis recomendado (destacado visualmente)
    ord_best : order recomendado (destacado visualmente)
    figsize  : tamaño de la figura
    save_path: ruta de guardado opcional
    dpi      : resolución al guardar

    Retorna
    -------
    fig : Figure
    """
    vr_pivot   = sel_df.pivot(index="order", columns="n_basis", values="var_retained")
    rmse_pivot = sel_df.pivot(index="order", columns="n_basis", values="rmse_mean")

    best_nb_idx  = list(vr_pivot.columns).index(nb_best)
    best_ord_idx = list(vr_pivot.index).index(ord_best)
    orders_list  = sorted(sel_df["order"].unique())

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Heatmaps VR y RMSE ───────────────────────────────────────────────────
    for ax, pivot, cmap, label in [
        (axes[0], vr_pivot * 100, "YlGn",     "Varianza retenida (%)"),
        (axes[1], rmse_pivot,     "YlOrRd_r", "RMSE medio de reconstrucción"),
    ]:
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("n_basis")
        ax.set_ylabel("order")
        ax.set_title(label)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=6.5)
        ax.add_patch(plt.Rectangle(
            (best_nb_idx - 0.5, best_ord_idx - 0.5), 1, 1,
            fill=False, edgecolor="crimson", lw=2.5, label="recomendado",
        ))
        ax.legend(fontsize=7, loc="upper left")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # ── Curva VR vs n_basis ──────────────────────────────────────────────────
    for i, ord_ in enumerate(orders_list):
        sub = sel_df[sel_df["order"] == ord_].sort_values("n_basis")
        axes[2].plot(
            sub["n_basis"], sub["var_retained"] * 100,
            marker="o", lw=1.4, ms=5,
            color=_COLORS_ORDER[i % len(_COLORS_ORDER)],
            label=f"order={ord_}",
        )
    axes[2].axvline(nb_best, color="crimson", ls="--", lw=1.2, label=f"nb_best={nb_best}")
    axes[2].set_xlabel("n_basis")
    axes[2].set_ylabel("Varianza retenida (%)")
    axes[2].set_title("Varianza retenida vs n_basis")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
