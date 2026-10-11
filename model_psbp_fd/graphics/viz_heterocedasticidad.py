"""
viz_heterocedasticidad.py
=========================
Figura del diagnostico banda contra varianza condicional (corrida 301). Solo dibuja:
los anchos los calcula `fit/heterocedasticidad.banda_contra_varianza_condicional`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

__all__ = ["plot_banda_contra_varianza"]

_AZUL, _ROJO = "#2c7fb8", "#c0392b"
_COLORES = {"equicolas": _AZUL, "simultanea": "#e08e0b", "hdr": "#2ca25f"}


def plot_banda_contra_varianza(e: Dict, ORIG: Dict, DIS: Dict, M: int,
                               save_path=None) -> plt.Figure:
    """Ancho integrado de la banda del PSBPM-FD y del oraculo en test: serie temporal y
    dispersion con la diagonal. Con `e["ancho_banda_psbp_intervalos"]` (varias
    construcciones, `banda_contra_varianza_condicional(..., intervalos=...)`) se
    superponen todas."""
    m = ~ORIG["es_train"]
    t, a_or = ORIG["t_orig"][m], e["ancho_banda_oraculo"][m]
    anchos = e.get("ancho_banda_psbp_intervalos") or {"equicolas": e["ancho_banda_psbp"]}
    varias = len(anchos) > 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.6, 1]})
    ax1.plot(t, a_or, lw=0.9, color="0.35", label="oráculo (h_t verdadera)")
    lim, resumen = [a_or.min(), a_or.max()], []
    for nombre, a_full in anchos.items():
        a = a_full[m]
        color = _COLORES.get(nombre, _AZUL) if varias else _AZUL
        etiqueta = f"PSBPM-FD ({nombre})" if varias else "PSBPM-FD"
        ax1.plot(t, a, lw=1.1 if nombre == "equicolas" else 0.9, color=color, label=etiqueta)
        ax2.scatter(a_or, a, s=9 if nombre == "equicolas" else 6, color=color, alpha=0.6, label=etiqueta if varias else None)
        lim = [min(lim[0], a.min()), max(lim[1], a.max())]
        resumen.append(f"{nombre + ': ' if varias else ''}Spearman {spearmanr(a, a_or)[0]:.3f} · "
                       f"CV {a.std() / a.mean():.3f}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("ancho integrado de la banda")
    ax1.legend(fontsize=8, loc="upper right")
    ax2.plot(lim, lim, color=_ROJO, lw=1.0, ls="--", label="y = x")
    ax2.set_xlabel("oráculo")
    ax2.set_ylabel("PSBPM-FD")
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"M={M} · ancho de la banda {DIS['NIVEL']:.0%} contra la varianza condicional "
                 f"verdadera (test) · " + " | ".join(resumen)
                 + f" · CV oráculo {a_or.std() / a_or.mean():.3f}", fontsize=10 if not varias else 9)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
