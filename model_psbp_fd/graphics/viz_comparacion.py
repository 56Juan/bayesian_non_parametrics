"""
viz_comparacion.py
==================
Figuras de la comparacion PSBPM-FD / FAR / RF sobre el barrido en M. Solo dibujan:
las cifras las calcula `fit/comparacion_barrido.py`.

Funciones publicas
------------------
ESTILOS_MODELOS         : color y trazo de cada modelo.
ESTILOS_BANDAS          : color de cada banda del Bloque B (FAR y construcciones del PSBPM-FD).
plot_ganador_modelo     : % de ventanas ganadas por modelo, train y test, para un M.
plot_metricas_vs_M      : promedio en test de cada metrica contra M, una linea por grupo.
plot_bandas_contraste   : dos bandas superpuestas sobre la misma curva, origenes extremos
                          (una HDR se dibuja por tramos).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .viz_comportamientos import ESTILOS_PREDICTORES

__all__ = ["ESTILOS_MODELOS", "ESTILOS_MODELOS_PREDICTORES", "ESTILOS_BANDAS", "plot_ganador_modelo",
           "plot_metricas_vs_M", "plot_bandas_contraste"]

# El FAR va grueso y punteado porque se TAPA con el PSBPM-FD cuando sus curvas casi coinciden.
ESTILOS_MODELOS = {
    "FAR":      dict(color="#1f6f8b", lw=3.0, ls="--"),
    "RF":       dict(color="#e08e0b", lw=1.4, ls="-"),
    "PSBPM-FD": dict(color="#c0392b", lw=1.6, ls="-", zorder=5),
}
# Una fila del PSBPM-FD por predictor puntual (`cargar_psbp(..., predictores=...)`).
ESTILOS_MODELOS_PREDICTORES = {
    "FAR": ESTILOS_MODELOS["FAR"], "RF": ESTILOS_MODELOS["RF"],
    **{f"PSBPM-FD ({p})": dict(est, zorder=5) for p, est in ESTILOS_PREDICTORES.items()},
}
# Las claves son G_FAR, G_PSG, G_PSB, G_PSS y G_PSH de `fit.comparacion_barrido`
# (tests/test_intervalos_predictiva.py verifica que coincidan).
ESTILOS_BANDAS = {
    "FAR (IC del modelo AR)": dict(color="#1f6f8b"),
    "PSBPM-FD (IC gaussiano, mismo mecanismo)": dict(color="#8c8c8c"),
    "PSBPM-FD (predictiva bayesiana)": dict(color="#c0392b"),
    "PSBPM-FD (banda simultánea)": dict(color="#e08e0b"),
    "PSBPM-FD (región de máxima densidad)": dict(color="#2ca25f"),
}
_AZUL, _ROJO = "#2c7fb8", "#c0392b"


def _tramos3(L, U):
    L, U = np.asarray(L, float), np.asarray(U, float)
    return (L, U) if L.ndim == 3 else (L[..., None], U[..., None])


def plot_ganador_modelo(gana: pd.DataFrame, M: int, metricas, w: int, save_path=None) -> plt.Figure:
    """Por metrica, el % de ventanas que gana cada modelo en un M, train contra test."""
    sub_M = gana[gana.M == M]
    modelos = sorted(sub_M["modelo"].unique())
    n = len(metricas)
    fig, axes = plt.subplots(1, n, figsize=(max(3.2, 1.1 * len(modelos) + 1.5) * n, 4.0),
                             sharey=True, squeeze=False)
    xs, ancho = np.arange(len(modelos)), 0.4
    for ax, (etiqueta, _c) in zip(axes[0], metricas):
        for i, (bloque, color) in enumerate((("train", _AZUL), ("test", _ROJO))):
            v = [float(sub_M[(sub_M.modelo == m) & (sub_M.bloque == bloque)
                             & (sub_M.metrica == etiqueta)]["pct_ventanas_ganadas"].iloc[0]) for m in modelos]
            barras = ax.bar(xs + (i - 0.5) * ancho, v, width=ancho, color=color, label=bloque)
            for b, val in zip(barras, v):
                if val >= 0.05:
                    ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.0%}", ha="center",
                            va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(modelos, rotation=30, ha="right", fontsize=8)
        ax.set_title(etiqueta, fontsize=9)
        ax.set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("% de ventanas ganadas")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"M={M} · qué modelo gana cada ventana (w={w})", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_metricas_vs_M(prom: pd.DataFrame, paneles: Sequence, grupos: Dict[str, dict],
                       titulo: str, nominal: float = None, save_path=None) -> plt.Figure:
    """
    Promedio en test de cada metrica contra M. `prom` es la salida de `promedio_vs_M`
    (columnas (columna, grupo)); `paneles` es `[(etiqueta, columna)]` y `grupos`
    `{nombre: estilo}` (color, y opcionalmente ls/lw). Con `nominal`, las metricas de
    cobertura llevan la linea del nivel nominal.
    """
    fig, axes = plt.subplots(1, len(paneles), figsize=(4.0 * len(paneles), 3.5), squeeze=False)
    for ax, (etiqueta, col) in zip(axes[0], paneles):
        for nombre, est in grupos.items():
            if (col, nombre) in prom.columns:
                ax.plot(prom.index, prom[(col, nombre)], marker="o", ms=5, lw=est.get("lw", 1.6),
                        ls=est.get("ls", "-"), color=est["color"], label=nombre)
        if nominal is not None and col.startswith("picp"):
            ax.axhline(nominal, color="k", ls=":", lw=1)
        ax.set_xticks(list(prom.index))
        ax.set_xlabel("M")
        ax.set_title(etiqueta, fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_bandas_contraste(bandas: Dict, extremos: Dict, X_obj: np.ndarray, grilla: np.ndarray,
                          t_orig: np.ndarray, M: int, metrica: str, nombres: Sequence[str],
                          grupo: str, save_path=None) -> plt.Figure:
    """
    Dos bandas superpuestas sobre la misma curva suavizada, en los origenes `grupo`
    ("mejores" o "peores") de `extremos`. `nombres` son los prefijos de las dos bandas
    de `bandas` a contrastar. El titulo de cada panel trae la cobertura puntual de
    cada una. La banda del FAR no depende del origen; la bayesiana si.
    """
    def _por_prefijo(pref):
        for n, v in bandas.items():
            if n.startswith(pref):
                return n, v
        return None, None
    n1, b1 = _por_prefijo(nombres[0])
    n2, b2 = _por_prefijo(nombres[1])
    if b1 is None or b2 is None:
        print(f"[M={M}] no estan las dos bandas {nombres}; se omite.")
        return None
    (L1, U1), Xp1 = _tramos3(*b1[0]), b1[1]
    (L2, U2), Xp2 = _tramos3(*b2[0]), b2[1]
    idx = extremos[grupo]
    fig, axes = plt.subplots(1, len(idx), figsize=(3.1 * len(idx), 3.3), sharey=True, squeeze=False)

    def _cob(i, L, U):
        y = X_obj[i][:, None]
        return float(np.mean(((y >= L[i]) & (y <= U[i])).any(axis=-1)))
    for ax, i in zip(axes[0], idx):
        for L, U, col in ((L1, U1, "#1f6f8b"), (L2, U2, "#c0392b")):
            for k in range(L.shape[-1]):        # una HDR: un relleno por tramo, con huecos NaN
                ax.fill_between(grilla, L[i, :, k], U[i, :, k], color=col, alpha=0.20, lw=0)
        ax.plot(grilla, Xp1[i], color="#1f6f8b", lw=1.3, ls="--")
        ax.plot(grilla, Xp2[i], color="#c0392b", lw=1.3)
        ax.plot(grilla, X_obj[i], color="k", lw=1.6)
        c1, c2 = _cob(i, L1, U1), _cob(i, L2, U2)
        ax.set_title(f"t={int(t_orig[i])}\ncob {c1:.2f} / {c2:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r"$\tau$", fontsize=8)
    axes[0, 0].set_ylabel(r"$X_t(\tau)$", fontsize=9)
    fig.legend(handles=[plt.Line2D([], [], color="k", lw=1.6, label="curva suavizada"),
                        plt.Line2D([], [], color="#1f6f8b", lw=1.3, ls="--", label=n1),
                        plt.Line2D([], [], color="#c0392b", lw=1.3, label=n2)],
               loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"M={M} · {len(idx)} orígenes {grupo} por {metrica} (test) · cobertura puntual en el título",
                 fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
