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
numpy, matplotlib
Las funciones de diagnóstico (autocorr, ess_geyer, geweke_z, gelman_rubin,
aggregate_trace, extract_var_trace) deben ser importadas o definidas en el
notebook. Este módulo las recibe como argumentos opcionales o las redefine
internamente si no se pasan.

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

def _autocorr(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """ACF muestral hasta max_lag (lag 0 = 1.0). Implementación FFT."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    x_c = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x_c, n=nfft)
    acf_full = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    acf_full /= acf_full[0]
    return acf_full[: max_lag + 1]


def _ess_geyer(x: np.ndarray) -> float:
    """Effective Sample Size con regla de truncamiento de Geyer (1992)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    rho = _autocorr(x, max_lag=min(n - 1, 1000))
    s = 0.0
    for k in range(1, len(rho) - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        s += pair
    tau_int = 1.0 + 2.0 * s
    return float(n / tau_int) if tau_int > 0 else float(n)


def _geweke_z(x: np.ndarray, first: float = 0.1, last: float = 0.5) -> float:
    """Diagnóstico de Geweke: test z entre segmentos de la cadena."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    n_a, n_b = int(first * n), int(last * n)
    if n_a < 2 or n_b < 2:
        return np.nan
    x_a, x_b = x[:n_a], x[-n_b:]
    var_a = x_a.var(ddof=1) / max(_ess_geyer(x_a), 1.0)
    var_b = x_b.var(ddof=1) / max(_ess_geyer(x_b), 1.0)
    denom = np.sqrt(var_a + var_b)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float((x_a.mean() - x_b.mean()) / denom)


def _gelman_rubin(chains: np.ndarray) -> float:
    """R̂ de Gelman-Rubin. chains shape: (m, n)."""
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError("chains debe ser 2D (m_chains, n_iter)")
    m, n = chains.shape
    if m < 2 or n < 2:
        return np.nan
    chain_means = chains.mean(axis=1)
    chain_vars  = chains.var(axis=1, ddof=1)
    B = n * chain_means.var(ddof=1)
    W = chain_vars.mean()
    if W == 0:
        return np.nan
    var_hat = (n - 1) / n * W + (1.0 / n) * B
    return float(np.sqrt(var_hat / W))


def _extract_var_trace(trace: np.ndarray, j: int, burn: int) -> np.ndarray:
    """Extrae traza post-burn para variable j. Admite (nsim,p) y (nsim,N,p)."""
    if trace.ndim == 3:
        return trace[burn:, :, j].mean(axis=1).astype(np.float64)
    elif trace.ndim == 2:
        return trace[burn:, j].astype(np.float64)
    else:
        raise ValueError(f"ndim={trace.ndim} no soportado")


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


def _compute_diag(chains_mat: np.ndarray) -> dict:
    """Calcula ESS, Geweke y R̂ sobre (n_chains, n_post)."""
    ess_c    = [_ess_geyer(chains_mat[c]) for c in range(chains_mat.shape[0])]
    geweke_c = [_geweke_z(chains_mat[c])  for c in range(chains_mat.shape[0])]
    rhat     = _gelman_rubin(chains_mat)
    return {
        "ess_min":      float(np.min(ess_c)),
        "ess_mean":     float(np.mean(ess_c)),
        "geweke_max":   float(np.max(np.abs(geweke_c))),
        "rhat":         rhat,
        "converge":     (rhat < 1.1 and np.min(ess_c) > 100
                         and np.max(np.abs(geweke_c)) < 2.0),
    }


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

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
) -> plt.Figure:
    """
    Trazas post-burn-in y ACF de p_j (pijout) por covariable del componente k.

    Mismos parámetros que `plot_traces_bj` (con trace_key='pijout').
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

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
) -> Tuple[plt.Figure, list]:
    """
    Panel completo de convergencia para β̄_j: traza | ACF | posterior.
    Incluye R̂, ESS_min y |Geweke|_max en el título de cada posterior.

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

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
) -> Tuple[plt.Figure, list]:
    """
    Panel completo de convergencia para p_j: traza | ACF | posterior.
    Incluye R̂, ESS_min y |Geweke|_max.

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

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, diag_records
