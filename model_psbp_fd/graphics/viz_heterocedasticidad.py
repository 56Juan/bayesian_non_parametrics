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


def plot_banda_contra_varianza(e: Dict, ORIG: Dict, DIS: Dict, M: int,
                               save_path=None) -> plt.Figure:
    """Ancho integrado de la banda del PSBPM-FD y del oraculo en test: serie temporal y
    dispersion con la diagonal."""
    m = ~ORIG["es_train"]
    t, a, a_or = ORIG["t_orig"][m], e["ancho_banda_psbp"][m], e["ancho_banda_oraculo"][m]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.6, 1]})
    ax1.plot(t, a_or, lw=0.9, color="0.35", label="oráculo (h_t verdadera)")
    ax1.plot(t, a, lw=1.1, color=_AZUL, label="PSBPM-FD")
    ax1.set_xlabel("t")
    ax1.set_ylabel("ancho integrado de la banda")
    ax1.legend(fontsize=8, loc="upper right")
    ax2.scatter(a_or, a, s=9, color=_AZUL, alpha=0.6)
    lim = [min(a.min(), a_or.min()), max(a.max(), a_or.max())]
    ax2.plot(lim, lim, color=_ROJO, lw=1.0, ls="--", label="y = x")
    ax2.set_xlabel("oráculo")
    ax2.set_ylabel("PSBPM-FD")
    ax2.legend(fontsize=8, loc="upper left")
    rho = spearmanr(a, a_or)[0]
    fig.suptitle(f"M={M} · ancho de la banda {DIS['NIVEL']:.0%} contra la varianza condicional "
                 f"verdadera (test) · Spearman {rho:.3f} · CV PSBPM-FD {a.std() / a.mean():.3f} "
                 f"/ oráculo {a_or.std() / a_or.mean():.3f}", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
