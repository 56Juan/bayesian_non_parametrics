"""
viz_traces.py
=============
Visualizadores de convergencia MCMC para el modelo PSBP-FD.

Funciones públicas
------------------
plot_traces_bj       : Trazas + ACF para β_j (betajhout) por variable.
plot_traces_pj       : Trazas + ACF para p_j (pijout) por variable.
plot_convergence_bj  : Convergencia + posterior β_j con métricas (ESS, R̂, Geweke).
plot_convergence_pj  : Convergencia + posterior p_j con métricas.

Dependencias
------------
numpy, matplotlib y `fit.diagnostics_mcmc`, que es donde viven los estadísticos
(ACF, ESS de Geyer, z de Geweke, R̂). Este módulo solo dibuja: para obtener la
tabla de diagnósticos sin generar figuras, usar `fit.tabla_diagnosticos`.

Convención de trazas
--------------------
betajhout : (nsim, N, p)   — β por átomo y variable
pijout    : (nsim, p)      — probabilidad de inclusión por variable
burn      : int            — iteraciones a descartar
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Diagnósticos estadísticos (autocontenidos)
# ─────────────────────────────────────────────────────────────────────────────

# Los estadisticos viven en `fit.diagnostics_mcmc`: una sola definicion de la
# ACF, el ESS de Geyer, el z de Geweke y R-hat en todo el proyecto. Aqui solo
# se les pone nombre local para no tocar el resto del modulo. Antes estaban
# definidos aqui, de modo que obtener la tabla de diagnosticos obligaba a
# generar las figuras.
from ..fit.diagnostics_mcmc import (
    autocorr as _autocorr,
    ess_geyer as _ess_geyer,
    geweke_z as _geweke_z,
    gelman_rubin as _gelman_rubin,
    extraer_traza_variable as _extract_var_trace,
    diagnostico_variable as _compute_diag,
)

# Progreso por consola. Estas cuatro funciones recorren p variables x n_chains
# calculando ACF (y, en los paneles de convergencia, ESS de Geyer y R-hat) antes
# de dibujar nada: con p grande la figura tarda en aparecer y no hay forma de
# saber si avanza. `verbose=True` lo hace visible sin alterar la figura.
from ..utils.progreso import Progreso


# ─────────────────────────────────────────────────────────────────────────────
# Paleta de colores para cadenas
# ─────────────────────────────────────────────────────────────────────────────
_CHAIN_COLORS = ["#1a6faf", "#e07b39", "#3aaa35", "#9b59b6", "#c0392b",
                 "#16a085", "#d35400", "#2c3e50"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _build_chains_matrix(
    models_chains: Dict,
    k: int,
    trace_key: str,
    j: int,
    burn: int,
    n_chains: int,
) -> np.ndarray:
    """Devuelve (n_chains, n_post) con trazas post-burn para variable j."""
    rows = []
    for c in range(n_chains):
        tr = models_chains[k][c].traces[trace_key]
        rows.append(_extract_var_trace(tr, j, burn))
    return np.vstack(rows)


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def plot_traces_bj(
    models_chains: Dict,
    k: int,
    burn: int,
    n_chains: int,
    feature_names: Optional[List[str]] = None,
    max_lag: int = 80,
    figsize_per_var: Tuple[float, float] = (14, 2.2),
    title_prefix: str = "",
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> plt.Figure:
    """
    Trazas post-burn-in y ACF de β̄_j (betajhout promediado sobre átomos)
    para cada covariable del componente k.

    Parámetros
    ----------
    models_chains : dict {k: {c: modelo}}
    k             : índice de componente (0-based)
    burn          : iteraciones de burn-in
    n_chains      : número de cadenas
    feature_names : lista de nombres de variables (len = p)
    max_lag       : lags máximos para ACF
    figsize_per_var : (ancho, alto) por fila de variable
    title_prefix  : prefijo opcional para el título
    save_path     : si se especifica, guarda la figura
    verbose       : imprime el avance por covariable. No altera la figura.

    Retorna
    -------
    fig : Figure
    """
    trace_key = "betajhout"
    p = models_chains[k][0].n_features_
    names = feature_names or [f"var_{j}" for j in range(p)]

    fig, axes = plt.subplots(
        p, 2,
        figsize=(figsize_per_var[0], figsize_per_var[1] * p),
        squeeze=False,
    )
    prefix = f"{title_prefix}  " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Trazas β̄_j — Componente {k + 1}",
        fontsize=11, y=1.002,
    )
    prog = Progreso(f"plot_traces_bj[FPC {k + 1}]", total=len(names),
                    verbose=verbose)

    for j, fname in enumerate(names):
        mat = _build_chains_matrix(models_chains, k, trace_key, j, burn, n_chains)
        n_post = mat.shape[1]

        ax_tr, ax_acf = axes[j]

        # ── Trazas superpuestas ──────────────────────────────────────────
        for c in range(n_chains):
            ax_tr.plot(mat[c], lw=0.6, alpha=0.75,
                       color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)],
                       label=f"cad {c + 1}")
        ax_tr.set_title(f"β̄_j  [{fname}]  — trazas", fontsize=8)
        ax_tr.set_xlabel("iteración post burn-in", fontsize=7)
        ax_tr.legend(fontsize=6, ncol=n_chains)
        ax_tr.grid(True, alpha=0.3)

        # ── ACF por cadena ───────────────────────────────────────────────
        ci = 1.96 / np.sqrt(n_post)
        for c in range(n_chains):
            rho = _autocorr(mat[c], max_lag=max_lag)
            ax_acf.plot(np.arange(max_lag + 1), rho, lw=0.8, alpha=0.8,
                        color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)])
        ax_acf.axhline(0,   color="k", lw=0.5)
        ax_acf.axhline( ci, color="k", ls=":", lw=0.7, label="±1.96/√n")
        ax_acf.axhline(-ci, color="k", ls=":", lw=0.7)
        ax_acf.set_title(f"β̄_j  [{fname}]  — ACF", fontsize=8)
        ax_acf.set_xlabel("lag", fontsize=7)
        ax_acf.set_ylim(-0.35, 1.05)
        ax_acf.legend(fontsize=6, loc="upper right")
        ax_acf.grid(True, alpha=0.3)

        prog.paso(f"{fname}  media={mat.mean():.4g}  n_post={n_post}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    prog.fin(f"figura {p}x2" + (f" -> {save_path}" if save_path else ""))
    return fig


def plot_traces_pj(
    models_chains: Dict,
    k: int,
    burn: int,
    n_chains: int,
    feature_names: Optional[List[str]] = None,
    max_lag: int = 80,
    figsize_per_var: Tuple[float, float] = (14, 2.2),
    title_prefix: str = "",
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> plt.Figure:
    """
    Trazas post-burn-in y ACF de p_j (pijout) por covariable del componente k.

    Mismos parámetros que `plot_traces_bj` (con trace_key='pijout'), incluido
    `verbose`.
    """
    trace_key = "pijout"
    p = models_chains[k][0].n_features_
    names = feature_names or [f"var_{j}" for j in range(p)]

    fig, axes = plt.subplots(
        p, 2,
        figsize=(figsize_per_var[0], figsize_per_var[1] * p),
        squeeze=False,
    )
    prefix = f"{title_prefix}  " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Trazas p_j — Componente {k + 1}",
        fontsize=11, y=1.002,
    )
    prog = Progreso(f"plot_traces_pj[FPC {k + 1}]", total=len(names),
                    verbose=verbose)

    for j, fname in enumerate(names):
        mat = _build_chains_matrix(models_chains, k, trace_key, j, burn, n_chains)
        n_post = mat.shape[1]
        ci = 1.96 / np.sqrt(n_post)

        ax_tr, ax_acf = axes[j]

        for c in range(n_chains):
            ax_tr.plot(mat[c], lw=0.6, alpha=0.75,
                       color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)],
                       label=f"cad {c + 1}")
        ax_tr.set_title(f"p_j  [{fname}]  — trazas", fontsize=8)
        ax_tr.set_xlabel("iteración post burn-in", fontsize=7)
        ax_tr.set_ylim(-0.05, 1.05)
        ax_tr.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax_tr.legend(fontsize=6, ncol=n_chains)
        ax_tr.grid(True, alpha=0.3)

        for c in range(n_chains):
            rho = _autocorr(mat[c], max_lag=max_lag)
            ax_acf.plot(np.arange(max_lag + 1), rho, lw=0.8, alpha=0.8,
                        color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)])
        ax_acf.axhline(0,   color="k", lw=0.5)
        ax_acf.axhline( ci, color="k", ls=":", lw=0.7, label="±1.96/√n")
        ax_acf.axhline(-ci, color="k", ls=":", lw=0.7)
        ax_acf.set_title(f"p_j  [{fname}]  — ACF", fontsize=8)
        ax_acf.set_xlabel("lag", fontsize=7)
        ax_acf.set_ylim(-0.35, 1.05)
        ax_acf.legend(fontsize=6, loc="upper right")
        ax_acf.grid(True, alpha=0.3)

        prog.paso(f"{fname}  PIP media={mat.mean():.4f}  n_post={n_post}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    prog.fin(f"figura {p}x2" + (f" -> {save_path}" if save_path else ""))
    return fig


def plot_convergence_bj(
    models_chains: Dict,
    k: int,
    burn: int,
    n_chains: int,
    feature_names: Optional[List[str]] = None,
    max_lag: int = 80,
    figsize_per_var: Tuple[float, float] = (15, 2.4),
    title_prefix: str = "",
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[plt.Figure, list]:
    """
    Panel completo de convergencia para β̄_j: traza | ACF | posterior.
    Incluye R̂, ESS_min y |Geweke|_max en el título de cada posterior.

    `verbose` imprime R̂, ESS y el veredicto de cada variable a medida que se
    calculan, en lugar de dejarlos escondidos en los títulos de una figura que
    aparece al final. No altera la figura ni `diag_records`.

    Retorna
    -------
    fig, diag_records : lista de dicts con métricas por variable
    """
    trace_key = "betajhout"
    p = models_chains[k][0].n_features_
    names = feature_names or [f"var_{j}" for j in range(p)]

    fig, axes = plt.subplots(
        p, 3,
        figsize=(figsize_per_var[0], figsize_per_var[1] * p),
        squeeze=False,
    )
    prefix = f"{title_prefix}  " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Convergencia β̄_j — Componente {k + 1}",
        fontsize=12, y=1.002,
    )
    prog = Progreso(f"plot_convergence_bj[FPC {k + 1}]", total=len(names),
                    verbose=verbose)

    diag_records = []
    for j, fname in enumerate(names):
        mat   = _build_chains_matrix(models_chains, k, trace_key, j, burn, n_chains)
        diag  = _compute_diag(mat)
        n_post = mat.shape[1]
        ci    = 1.96 / np.sqrt(n_post)
        diag_records.append({"variable": fname, **diag})

        ax_tr, ax_acf, ax_post = axes[j]

        # Trazas
        for c in range(n_chains):
            ax_tr.plot(mat[c], lw=0.6, alpha=0.75,
                       color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)],
                       label=f"cad {c + 1}")
        ax_tr.set_title(f"β̄_j [{fname}]  trazas", fontsize=8)
        ax_tr.set_xlabel("iter post burn-in", fontsize=7)
        ax_tr.legend(fontsize=6, ncol=n_chains)
        ax_tr.grid(True, alpha=0.3)

        # ACF
        for c in range(n_chains):
            rho = _autocorr(mat[c], max_lag=max_lag)
            ax_acf.plot(np.arange(max_lag + 1), rho, lw=0.8, alpha=0.8,
                        color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)])
        ax_acf.axhline(0,   color="k", lw=0.5)
        ax_acf.axhline( ci, color="k", ls=":", lw=0.7)
        ax_acf.axhline(-ci, color="k", ls=":", lw=0.7)
        ax_acf.set_title(f"β̄_j [{fname}]  ACF", fontsize=8)
        ax_acf.set_xlabel("lag", fontsize=7)
        ax_acf.set_ylim(-0.35, 1.05)
        ax_acf.grid(True, alpha=0.3)

        # Posterior
        combined = mat.ravel()
        ax_post.hist(combined, bins=40, color="steelblue", alpha=0.75,
                     edgecolor="white")
        ax_post.axvline(combined.mean(), color="k", lw=1.2, ls="--",
                        label=f"media={combined.mean():.3f}")
        conv_flag = "✓" if diag["converge"] else "✗"
        ax_post.set_title(
            f"β̄_j [{fname}]  "
            f"R̂={diag['rhat']:.3f}  ESS={diag['ess_min']:.0f}  "
            f"|G|={diag['geweke_max']:.2f}  {conv_flag}",
            fontsize=8,
        )
        ax_post.legend(fontsize=6)
        ax_post.grid(True, alpha=0.3)

        prog.paso(f"{fname}  rhat={diag['rhat']:.3f}  "
                  f"ess_min={diag['ess_min']:.0f}  "
                  f"|G|={diag['geweke_max']:.2f}  {conv_flag}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    prog.fin(f"{sum(not d['converge'] for d in diag_records)}/{p} sin converger")
    return fig, diag_records


def plot_convergence_pj(
    models_chains: Dict,
    k: int,
    burn: int,
    n_chains: int,
    feature_names: Optional[List[str]] = None,
    max_lag: int = 80,
    figsize_per_var: Tuple[float, float] = (15, 2.4),
    title_prefix: str = "",
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[plt.Figure, list]:
    """
    Panel completo de convergencia para p_j: traza | ACF | posterior.
    Incluye R̂, ESS_min y |Geweke|_max.

    `verbose` se comporta como en `plot_convergence_bj`.

    Retorna
    -------
    fig, diag_records
    """
    trace_key = "pijout"
    p = models_chains[k][0].n_features_
    names = feature_names or [f"var_{j}" for j in range(p)]

    fig, axes = plt.subplots(
        p, 3,
        figsize=(figsize_per_var[0], figsize_per_var[1] * p),
        squeeze=False,
    )
    prefix = f"{title_prefix}  " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Convergencia p_j — Componente {k + 1}",
        fontsize=12, y=1.002,
    )
    prog = Progreso(f"plot_convergence_pj[FPC {k + 1}]", total=len(names),
                    verbose=verbose)

    diag_records = []
    for j, fname in enumerate(names):
        mat   = _build_chains_matrix(models_chains, k, trace_key, j, burn, n_chains)
        diag  = _compute_diag(mat)
        n_post = mat.shape[1]
        ci    = 1.96 / np.sqrt(n_post)
        diag_records.append({"variable": fname, **diag})

        ax_tr, ax_acf, ax_post = axes[j]

        for c in range(n_chains):
            ax_tr.plot(mat[c], lw=0.6, alpha=0.75,
                       color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)],
                       label=f"cad {c + 1}")
        ax_tr.set_title(f"p_j [{fname}]  trazas", fontsize=8)
        ax_tr.set_xlabel("iter post burn-in", fontsize=7)
        ax_tr.set_ylim(-0.05, 1.05)
        ax_tr.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6, label="0.5")
        ax_tr.legend(fontsize=6, ncol=n_chains)
        ax_tr.grid(True, alpha=0.3)

        for c in range(n_chains):
            rho = _autocorr(mat[c], max_lag=max_lag)
            ax_acf.plot(np.arange(max_lag + 1), rho, lw=0.8, alpha=0.8,
                        color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)])
        ax_acf.axhline(0,   color="k", lw=0.5)
        ax_acf.axhline( ci, color="k", ls=":", lw=0.7)
        ax_acf.axhline(-ci, color="k", ls=":", lw=0.7)
        ax_acf.set_title(f"p_j [{fname}]  ACF", fontsize=8)
        ax_acf.set_xlabel("lag", fontsize=7)
        ax_acf.set_ylim(-0.35, 1.05)
        ax_acf.grid(True, alpha=0.3)

        combined = mat.ravel()
        ax_post.hist(combined, bins=40, color="darkorange", alpha=0.75,
                     edgecolor="white")
        ax_post.axvline(combined.mean(), color="k", lw=1.2, ls="--",
                        label=f"media={combined.mean():.3f}")
        ax_post.axvline(0.5, color="gray", ls=":", lw=1.0, label="0.5")
        conv_flag = "✓" if diag["converge"] else "✗"
        ax_post.set_title(
            f"p_j [{fname}]  "
            f"R̂={diag['rhat']:.3f}  ESS={diag['ess_min']:.0f}  "
            f"|G|={diag['geweke_max']:.2f}  {conv_flag}",
            fontsize=8,
        )
        ax_post.legend(fontsize=6)
        ax_post.grid(True, alpha=0.3)

        prog.paso(f"{fname}  rhat={diag['rhat']:.3f}  "
                  f"ess_min={diag['ess_min']:.0f}  "
                  f"|G|={diag['geweke_max']:.2f}  {conv_flag}")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    prog.fin(f"{sum(not d['converge'] for d in diag_records)}/{p} sin converger")
    return fig, diag_records
