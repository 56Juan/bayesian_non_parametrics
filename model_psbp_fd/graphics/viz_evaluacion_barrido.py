"""
viz_evaluacion_barrido.py
=========================
Figuras de la evaluacion sobre el barrido en M. Solo dibujan: las cifras las
calcula `fit/evaluacion_barrido.py`.

Funciones publicas
------------------
plot_ganador_ventana    : % de ventanas ganadas por cada M, train y test.
plot_scores_banda       : serie de cada score con su banda de credibilidad.
plot_scores_dispersion  : xi_hat contra xi, un panel por componente.

Las dos ultimas arman una REJILLA (`ncols` columnas) en vez de una sola fila o
columna: con M hasta 10 la figura de una fila pasaba de 40 pulgadas.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["plot_ganador_ventana", "plot_scores_banda", "plot_scores_dispersion"]

_AZUL, _ROJO = "#2c7fb8", "#c0392b"


def _rejilla(n: int, ncols: int, ancho_panel: float, alto_panel: float, **kw):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, min(n, ncols), squeeze=False,
                             figsize=(ancho_panel * min(n, ncols), alto_panel * nrows + 0.8), **kw)
    for j in range(n, nrows * min(n, ncols)):
        axes[j // ncols, j % ncols].set_visible(False)
    return fig, axes


def plot_ganador_ventana(gana: pd.DataFrame, metricas, w: int, M_OK: Sequence[int],
                         titulo: str, save_path=None) -> plt.Figure:
    """Por metrica, el % de ventanas que gana cada M, en train y en test. Cada
    barra lleva su valor; el eje y es comun para poder comparar paneles."""
    n = len(metricas)
    fig, axes = plt.subplots(1, n, figsize=(max(4.0, 0.55 * len(M_OK) * 2 + 1.5) * n, 4.2),
                             sharey=True, squeeze=False)
    xs, ancho = np.arange(len(M_OK)), 0.4
    for ax, (etiqueta, _) in zip(axes[0], metricas):
        sub = gana[(gana.metrica == etiqueta) & (gana.w == w)]
        for i, (bloque, color) in enumerate((("train", _AZUL), ("test", _ROJO))):
            v = [float(sub[(sub.M == M) & (sub.bloque == bloque)]["pct_ventanas_ganadas"].iloc[0])
                 for M in M_OK]
            barras = ax.bar(xs + (i - 0.5) * ancho, v, width=ancho, color=color, label=bloque)
            for b, val in zip(barras, v):
                if val >= 0.05:
                    ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.0%}", ha="center",
                            va="bottom", fontsize=6.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"M={M}" for M in M_OK], rotation=45 if len(M_OK) > 5 else 0, fontsize=8)
        ax.set_title(etiqueta, fontsize=9)
        ax.set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("% de ventanas ganadas")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_scores_banda(e: Dict, ORIG: Dict, DIS: Dict, M: int, ncols: int = 2,
                      save_path=None) -> plt.Figure:
    """
    Serie de cada score con su banda de credibilidad. La zona de test va
    sombreada y los orígenes cuyo score cae FUERA de la banda van marcados en
    rojo, de modo que se ve donde falla la banda y no solo cuanto.
    """
    t, T0, NIVEL, es_train = ORIG["t_orig"], DIS["T0"], DIS["NIVEL"], ORIG["es_train"]
    n = e["n_components"]
    fig, axes = _rejilla(n, ncols, 7.0, 2.1, sharex=True)
    for k in range(n):
        ax = axes[k // ncols, k % ncols]
        y, yh, lo, hi = e["Y_obs"][:, k], e["Y_hat"][:, k], e["li_s"][:, k], e["ls_s"][:, k]
        fuera = (y < lo) | (y > hi)
        ax.axvspan(T0, t[-1], color="0.93", zorder=0)
        ax.fill_between(t, lo, hi, color=_AZUL, alpha=0.25, label=f"IC {NIVEL:.0%}")
        ax.plot(t, y, lw=0.8, color="0.25", label=r"$\xi$")
        ax.plot(t, yh, lw=0.9, color=_ROJO, label=r"$\hat\xi$")
        ax.scatter(t[fuera], y[fuera], s=9, color=_ROJO, zorder=4, label="fuera de la banda")
        ax.axvline(T0, color="crimson", ls="--", lw=1.0)
        ax.set_ylabel(rf"$\xi_{{{e['component_idx'][k] + 1}}}$")
        ax.set_title(f"PICP train {1 - fuera[es_train].mean():.3f} / test {1 - fuera[~es_train].mean():.3f}"
                     f" (nominal {NIVEL:.2f})", fontsize=8, loc="left")
    axes[0, 0].legend(fontsize=7, ncol=4, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("t")
    fig.suptitle(f"M={M} · banda de credibilidad de cada score (T0={T0}, nivel {NIVEL:.0%})",
                 fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_scores_dispersion(e: Dict, ORIG: Dict, DIS: Dict, M: int, cada_barra: int = 12,
                           ncols: int = 4, save_path=None) -> plt.Figure:
    """
    xi_hat contra xi con la diagonal, train y test con marcador distinto. Nube
    pegada a la diagonal = prediccion; nube horizontal = media incondicional;
    pendiente < 1 = encogimiento (esperado en un predictor bayesiano). Las barras
    son el IC de una submuestra de origenes.
    """
    es_train, NIVEL = ORIG["es_train"], DIS["NIVEL"]
    n = e["n_components"]
    fig, axes = _rejilla(n, ncols, 4.2, 4.4)
    for k in range(n):
        ax = axes[k // ncols, k % ncols]
        y, yh, lo, hi = e["Y_obs"][:, k], e["Y_hat"][:, k], e["li_s"][:, k], e["ls_s"][:, k]
        for m, nombre, color, marca in ((es_train, "train", _AZUL, "o"), (~es_train, "test", _ROJO, "^")):
            ax.scatter(y[m], yh[m], s=9, alpha=0.45, c=color, marker=marca, linewidths=0, label=nombre)
        sel = np.arange(0, ORIG["n_orig"], cada_barra)
        ax.vlines(y[sel], lo[sel], hi[sel], color="0.5", lw=0.7, alpha=0.6, zorder=0)
        lim = [float(min(y.min(), lo.min())), float(max(y.max(), hi.max()))]
        ax.plot(lim, lim, "k--", lw=1.1, zorder=3, label="$y=x$")
        pend, inter = np.polyfit(y, yh, 1)
        ax.plot(lim, [pend * lim[0] + inter, pend * lim[1] + inter], color="#2ca25f", lw=1.2,
                zorder=3, label=f"ajuste ({pend:.2f})")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", "box")
        te = ~es_train
        r2 = 1.0 - ((y - yh) ** 2)[te].sum() / max(float(((y[te] - y[te].mean()) ** 2).sum()), 1e-12)
        cob = float(((y >= lo) & (y <= hi))[te].mean())
        ax.set_title(f"FPC {e['component_idx'][k] + 1}\n$R^2$ test {r2:+.3f} · pend. {pend:.2f} · "
                     f"PICP test {cob:.3f}", fontsize=8.5)
        ax.set_xlabel(rf"$\xi_{{{e['component_idx'][k] + 1}}}$ observado", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel(r"$\hat\xi$ (media predictiva)", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(rf"M={M} · $\hat\xi$ contra $\xi$ · barras = IC {NIVEL:.0%} (una de cada {cada_barra})",
                 fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
