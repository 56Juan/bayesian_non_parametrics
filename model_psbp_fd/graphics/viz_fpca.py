"""
viz_fpca.py
===========
Visualizadores de FPCA y análisis de rezagos para PSBP-FD.

Funciones públicas
------------------
plot_fpca_scree               : Scree plot (log) + varianza acumulada.
plot_fpca_correlacion_lag0    : Heatmap de correlación contemporánea R₀ entre scores.
plot_rezagos_heatmap          : Heatmap de correlación (Pearson o Spearman) entre
                                respuesta en t y scores rezagados.
plot_series_componentes       : Una serie temporal por componente (coeficientes o
                                scores), en rejilla compacta.
plot_diagnostico_rezagos      : Los dos heatmaps (Pearson y Spearman) de respuesta vs
                                rezagos que arma `fit.preparacion_barrido`.

Contexto
--------
Estas funciones cubren §3.2 (diagnóstico FPCA), §3.3 (validación de scores)
y §4.1 (análisis de rezagos) del notebook 01_02_modelo.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos de estilo
# ─────────────────────────────────────────────────────────────────────────────

def _text_style(val: float) -> Tuple[int, str, str, Optional[str]]:
    """Devuelve (fontsize, fontweight, fontstyle, bbox_color) según |r|."""
    a = abs(val)
    if   a >= 0.7: return 10, "bold",   "normal", "gold"
    elif a >= 0.5: return  9, "bold",   "normal", "#cccccc"
    elif a >= 0.3: return  9, "normal", "italic", None
    else:          return  8, "normal", "normal", None


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def plot_fpca_scree(
    evals: np.ndarray,
    var_cum: np.ndarray,
    M_sugerido: int,
    var_target: float = 0.99,
    figsize: Tuple[float, float] = (16, 4),
    save_path: Optional[str] = None,
    dpi: int = 110,
) -> plt.Figure:
    """
    Scree plot (escala log) + varianza acumulada, con línea vertical en M_sugerido.

    Parámetros
    ----------
    evals       : (K,) autovalores ordenados de mayor a menor
    var_cum     : (K,) varianza acumulada proporcional (cumsum de var_ratio)
    M_sugerido  : nº de componentes sugerido (referencia visual)
    var_target  : umbral de varianza para la línea horizontal (default 0.99)
    figsize     : tamaño de la figura
    save_path   : ruta de guardado opcional
    dpi         : resolución al guardar

    Retorna
    -------
    fig : Figure
    """
    K = len(evals)
    componentes = np.arange(1, K + 1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    # ── Scree ────────────────────────────────────────────────────────────────
    ax0.plot(componentes, evals, "o-", lw=1.4, ms=5, color="steelblue")
    ax0.axvline(M_sugerido, ls="--", c="crimson", lw=1.2,
                label=f"sug. M={M_sugerido}")
    ax0.set_title("Scree (λ_m)")
    ax0.set_xlabel("componente")
    ax0.set_ylabel("λ_m")
    ax0.set_yscale("log")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.35)

    # ── Varianza acumulada ───────────────────────────────────────────────────
    ax1.plot(componentes, var_cum, "o-", lw=1.4, ms=5, color="steelblue")
    ax1.axhline(var_target, ls=":", c="grey", lw=1.0,
                label=f"target={var_target:.0%}")
    ax1.axvline(M_sugerido, ls="--", c="crimson", lw=1.2,
                label=f"sug. M={M_sugerido}")
    ax1.set_title("Varianza acumulada")
    ax1.set_xlabel("componente")
    ax1.set_ylabel("proporción")
    ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.35)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_fpca_correlacion_lag0(
    R0: np.ndarray,
    figsize: Tuple[float, float] = (5.2, 4.4),
    save_path: Optional[str] = None,
    dpi: int = 110,
) -> plt.Figure:
    """
    Heatmap de la matriz de correlación contemporánea R₀ (lag 0) entre scores FPCA.

    Útil para verificar que las componentes son efectivamente no correlacionadas
    en t (condición de ortogonalidad en L²).

    Parámetros
    ----------
    R0        : (M, M) matriz de correlación de los scores ξ en lag 0
    figsize   : tamaño de la figura
    save_path : ruta de guardado opcional
    dpi       : resolución al guardar

    Retorna
    -------
    fig : Figure
    """
    M = R0.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(R0, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(r"$R_0$: correlación contemporánea (lag 0)", fontsize=11)
    ax.set_xlabel("FPC")
    ax.set_ylabel("FPC")
    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    ax.set_xticklabels(range(1, M + 1))
    ax.set_yticklabels(range(1, M + 1))
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_rezagos_heatmap(
    matrix: np.ndarray,
    col_labels: List[str],
    row_labels: List[str],
    title: str,
    n_lags_max: int,
    K_total: int,
    band: float,
    vclip: float = 0.6,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    dpi: int = 110,
) -> plt.Figure:
    """
    Heatmap de correlación (Pearson o Spearman) entre la respuesta ξ(t) y los
    predictores ξ(t-lag) para todos los lags y componentes.

    Incluye:
    - Anotaciones de valor con estilo según |r| (cuadro dorado ≥ 0.7, gris ≥ 0.5).
    - Separadores verticales entre bloques de lag.
    - Leyenda de niveles de correlación y banda de significancia.

    Parámetros
    ----------
    matrix     : (K_total, K_total * n_lags_max) matriz de correlaciones
    col_labels : etiquetas de columnas (predictores), longitud K_total * n_lags_max
    row_labels : etiquetas de filas (respuestas), longitud K_total
    title      : título de la figura
    n_lags_max : número de lags evaluados (para dibujar separadores)
    K_total    : número de componentes FPCA
    band       : banda de significancia ±1.96/√n (referencia visual)
    vclip      : saturación simétrica del colormap (default 0.6)
    figsize    : tamaño de la figura; si None, se calcula automáticamente
    save_path  : ruta de guardado opcional
    dpi        : resolución al guardar

    Retorna
    -------
    fig : Figure
    """
    n_cov = len(col_labels)
    if figsize is None:
        figsize = (0.9 * n_cov + 2, 1.4 * K_total + 1.2)

    norm = mcolors.Normalize(vmin=-vclip, vmax=vclip, clip=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(n_cov))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(K_total))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("predictor (FPC, lag)")
    ax.set_ylabel("respuesta (t)")
    ax.set_title(title, fontsize=11)

    # ── Separadores entre bloques de lag ─────────────────────────────────────
    for lag in range(1, n_lags_max):
        ax.axvline(lag * K_total - 0.5, color="black", lw=1.2)

    # ── Anotaciones de valor ──────────────────────────────────────────────────
    for i in range(K_total):
        for j in range(n_cov):
            val = matrix[i, j]
            fs, fw, fst, box_color = _text_style(val)
            bbox_kw = (
                dict(boxstyle="round,pad=0.15", facecolor=box_color,
                     edgecolor="dimgray", alpha=0.85, lw=0.8)
                if box_color else None
            )
            ax.text(
                j, i, f"{val:+.2f}",
                ha="center", va="center",
                color="white" if abs(val) > 0.35 else "black",
                fontsize=fs, fontweight=fw, fontstyle=fst,
                bbox=bbox_kw,
            )

    # ── Leyenda ───────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="gold",    edgecolor="dimgray", label="|r| ≥ 0.7 (alta)"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="dimgray", label="|r| ≥ 0.5 (mod-alta)"),
        mpatches.Patch(facecolor="white",   edgecolor="white",   label="|r| ≥ 0.3 (mod)"),
        mpatches.Patch(facecolor="white",   edgecolor="white",   label=f"banda ±{band:.2f}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7.5,
              framealpha=0.9, title="Nivel", title_fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label=f"correlación (sat. ±{vclip})")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_series_componentes(
    Y: np.ndarray,
    T0: Optional[int] = None,
    simbolo: str = r"\xi",
    titulo: Optional[str] = None,
    ncols: int = 2,
    altura_panel: float = 1.5,
    ancho: float = 14.0,
    color: str = "C0",
    save_path: Optional[str] = None,
    dpi: int = 110,
) -> plt.Figure:
    r"""
    Una serie temporal por columna de `Y` (coeficientes o scores), en una
    rejilla de `ncols` columnas para que la figura no se alargue.

    Parámetros
    ----------
    Y            : (T, K) una columna por componente
    T0           : corte train/test; si se da, se marca y el título de cada panel
                   trae sd y autocorrelación a rezago 1 del bloque de entrenamiento
    simbolo      : símbolo de la etiqueta (LaTeX), p. ej. r"\theta" o r"\xi"
    ncols        : columnas de la rejilla
    altura_panel : alto de cada panel, en pulgadas
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    T, K = Y.shape
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False,
                             figsize=(ancho, altura_panel * nrows + 0.8))
    for j in range(nrows * ncols):
        ax = axes[j // ncols, j % ncols]
        if j >= K:
            ax.set_visible(False)
            continue
        y = Y[:, j]
        ax.plot(np.arange(T), y, lw=0.8, color=color)
        ref = y[:T0] if T0 else y
        ax.axhline(ref.mean(), color="0.6", lw=0.6)
        if T0:
            ax.axvline(T0, color="crimson", ls="--", lw=1.0)
        ax.set_ylabel(rf"${simbolo}_{{{j + 1}}}$")
        c = ref - ref.mean()
        ar1 = float((c[1:] * c[:-1]).sum() / max((c[:-1] ** 2).sum(), 1e-300))
        ax.set_title(f"sd {ref.std():.3f}" + (f" · ar1 {ar1:+.2f}" if T0 else ""),
                     fontsize=8, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    if titulo:
        fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig


def plot_diagnostico_rezagos(diag: dict, M: int, save_dir=None, vclip: float = 0.6):
    """
    Heatmaps de Pearson y de Spearman entre la respuesta en t y los scores rezagados, a
    partir de `diagnostico_rezagos` (solo train). Guarda `09a_rezagos_pearson.png` y
    `09b_rezagos_spearman.png` en `save_dir` si se da. Retorna las dos figuras.
    """
    from pathlib import Path
    figs = []
    for matriz, nombre, arch in ((diag["corr_pearson"], "Pearson", "09a"),
                                 (diag["corr_spearman"], "Spearman", "09b")):
        figs.append(plot_rezagos_heatmap(
            matriz, diag["col_labels"], diag["row_labels"],
            title=f"{nombre} — respuesta($t$) vs rezagos 1..{diag['n_lags_max']}  (M={M})",
            n_lags_max=diag["n_lags_max"], K_total=diag["K_total"], band=diag["band"], vclip=vclip,
            save_path=(str(Path(save_dir) / f"{arch}_rezagos_{nombre.lower()}.png") if save_dir else None)))
    return figs
