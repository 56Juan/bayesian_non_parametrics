"""
viz_convergencia.py
===================
Figuras del analisis de convergencia sobre el barrido en M. Solo dibujan: las
cifras las calcula `fit/convergencia_barrido.py`. Cada funcion devuelve la
figura y, si se da `save_path`, la guarda.

Funciones publicas
------------------
plot_ocupacion_mezcla   : traza e histograma de atomos ocupados por componente.
plot_gating             : senal de gating y sd del argumento probit.
plot_diagnostico_vs_M   : R-hat, ESS, Geweke y ocupacion a lo largo de M.
plot_ocupacion_vs_M     : ocupacion de la mezcla por M y componente.
plot_pip_vs_M           : PIP por covariable a lo largo de M.
plot_variantes          : ESS y R-hat por variable, variantes lado a lado.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "plot_ocupacion_mezcla", "plot_gating", "plot_diagnostico_vs_M",
    "plot_ocupacion_vs_M", "plot_pip_vs_M", "plot_variantes",
]


def _guardar(fig, save_path, dpi: int = 150):
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def plot_ocupacion_mezcla(e: Dict, M: int, save_path=None) -> plt.Figure:
    """Atomos ocupados (`N1out`) por iteracion y su histograma post-burn."""
    ci, n_comp, N, burn = e["component_idx"], e["n_components"], e["n_atomos"], e["burn"]
    fig, axes = plt.subplots(n_comp, 2, figsize=(13, 2.6 * n_comp), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    for k in range(n_comp):
        ax_tr, ax_hi = axes[k]
        for c in sorted(e["models_chains"][k]):
            n1 = e["models_chains"][k][c].traces["N1out"]
            ax_tr.plot(np.arange(len(n1)), n1, lw=0.7, alpha=0.75, label=f"cadena {c+1}")
            ax_hi.hist(n1[burn:], bins=np.arange(0.5, N + 1.5), alpha=0.55,
                       orientation="horizontal")
        ax_tr.axvline(burn, color="k", ls="--", lw=1, alpha=0.7)
        ax_tr.axhline(N, color="#c0392b", ls=":", lw=1.2)
        ax_tr.set_ylabel(f"FPC {ci[k]+1}\nátomos ocupados")
        ax_tr.set_ylim(0, N + 1)
        ax_hi.set_ylim(0, N + 1)
        ax_hi.set_xlabel("frec. post-burn")
        if k == 0:
            ax_tr.legend(fontsize=8, ncol=3)
            ax_tr.text(burn, N, " burn", fontsize=8, va="top")
    axes[-1, 0].set_xlabel("iteración")
    fig.suptitle(f"M={M} - ocupación de la mezcla · truncamiento N={N}", fontsize=12)
    return _guardar(fig, save_path)


def plot_gating(series: Dict, e: Dict, M: int, save_path=None) -> plt.Figure:
    """Senal de gating (atomos global - por punto) y sd del argumento probit."""
    ci, n_comp = e["component_idx"], e["n_components"]
    fig, axes = plt.subplots(n_comp, 2, figsize=(13, 2.6 * n_comp), squeeze=False)
    for k in range(n_comp):
        for c, S in series[k].items():
            axes[k, 0].plot(S["senal_gating"], lw=0.7, label=f"cadena {c+1}")
            axes[k, 1].plot(S["sd_argumento"], lw=0.7, label=f"cadena {c+1}")
        axes[k, 0].axhline(0, color="k", lw=0.8)
        axes[k, 0].set_ylabel(f"FPC {ci[k]+1}")
        if k == 0:
            axes[k, 0].set_title("señal de gating (átomos global - por punto)")
            axes[k, 1].set_title("sd del argumento probit")
            axes[k, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel("iteración post-burn")
    axes[-1, 1].set_xlabel("iteración post-burn")
    fig.suptitle(f"M={M} - gating", fontsize=12)
    return _guardar(fig, save_path)


def plot_diagnostico_vs_M(barrido: pd.DataFrame, umbrales: Dict, titulo: str,
                          save_path=None) -> plt.Figure:
    """R-hat max, ESS min, |Geweke| max y ocupacion media contra M; los umbrales
    van como linea horizontal para que la lectura no dependa de recordarlos."""
    Ms = list(barrido.index)
    paneles = [("rhat_max", "Rhat máximo", umbrales["rhat"], "menor es mejor"),
               ("ess_min", "ESS mínimo", umbrales["ess"], "mayor es mejor"),
               ("geweke_max", "|Geweke| máximo", umbrales["geweke"], "menor es mejor"),
               ("ocupacion_media", "átomos ocupados (med.)", None, "descriptivo")]
    fig, axes = plt.subplots(1, len(paneles), figsize=(4.0 * len(paneles), 3.4), squeeze=False)
    for ax, (col, nombre, umbral, nota) in zip(axes[0], paneles):
        ax.plot(Ms, barrido[col].to_numpy(dtype=float), "o-", lw=1.8, ms=7, color="#2c7fb8")
        if umbral is not None:
            ax.axhline(umbral, color="#c0392b", ls="--", lw=1.2, label=f"umbral {umbral}")
            ax.legend(fontsize=8)
        if col == "ocupacion_media":
            ax.axhline(barrido["N_trunc"].iloc[0], color="#c0392b", ls=":", lw=1.2)
        ax.set_xticks(Ms)
        ax.set_xlabel("M (componentes FPCA)")
        ax.set_title(f"{nombre}\n({nota})", fontsize=10)
    fig.suptitle(f"{titulo} - diagnóstico de muestreo vs M", fontsize=12)
    return _guardar(fig, save_path)


def plot_ocupacion_vs_M(piv: pd.DataFrame, N_trunc: int, save_path=None) -> plt.Figure:
    """Ocupacion media por componente a lo largo de M: dice si al subir M el modelo
    parte la mezcla en mas atomos. Se separa por FPC porque el promedio global
    esconde que sean las componentes NUEVAS las que se fragmentan."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for fpc in piv.columns:
        s = piv[fpc].dropna()
        ax.plot(s.index, s.to_numpy(), "o-", lw=1.6, ms=6, label=f"FPC {int(fpc)}")
    ax.axhline(N_trunc, color="#c0392b", ls=":", lw=1.2, label=f"truncamiento N={N_trunc}")
    ax.set_xticks(list(piv.index))
    ax.set_xlabel("M (componentes FPCA)")
    ax.set_ylabel("átomos ocupados (media post-burn)")
    ax.set_title("Fragmentación de la mezcla a lo largo del barrido")
    ax.legend(fontsize=8, ncol=2)
    return _guardar(fig, save_path)


def plot_pip_vs_M(pip_largo: pd.DataFrame, M_OK: Sequence[int], save_path=None) -> plt.Figure:
    """PIP global por covariable (media sobre componentes) contra M. Una covariable
    presente en todos los M deberia mantener su PIP; las que solo existen desde
    cierto M aparecen con la linea cortada."""
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for cov, g in pip_largo.groupby("covariable"):
        s = g.groupby("M")["pip"].mean()
        ax.plot(s.index, s.to_numpy(), "o-", lw=1.5, ms=6, label=str(cov))
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.6, label="umbral 0.5")
    ax.set_xticks(list(M_OK))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("M (componentes FPCA)")
    ax.set_ylabel("PIP global (media sobre componentes)")
    ax.set_title("Probabilidad de inclusión por covariable a lo largo del barrido", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    return _guardar(fig, save_path)


def plot_variantes(largo: pd.DataFrame, variantes: Sequence[str], titulo: str,
                   save_path=None) -> plt.Figure:
    """ESS min (beta_j y p_j) y R-hat (p_j) por variable, variantes lado a lado."""
    paneles = [("beta_j", "ess_min", "ESS mín · beta_j", 100),
               ("p_j", "ess_min", "ESS mín · p_j", 100),
               ("p_j", "rhat", "Rhat · p_j", 1.1)]
    fig, axes = plt.subplots(len(paneles), 1, figsize=(12, 2.8 * len(paneles)), squeeze=False)
    for ax, (par, col, nombre, umbral) in zip(axes[:, 0], paneles):
        piv = largo[largo.param == par].pivot_table(
            index="etiqueta", columns="variante", values=col, sort=False)[list(variantes)]
        piv.plot.bar(ax=ax, width=0.8)
        ax.axhline(umbral, color="#c0392b", ls="--", lw=1)
        ax.set_title(nombre, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        if col == "ess_min":
            ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=len(variantes))
    fig.suptitle(titulo, fontsize=12)
    return _guardar(fig, save_path)
