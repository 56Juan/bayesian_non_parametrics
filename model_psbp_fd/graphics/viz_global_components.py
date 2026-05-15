"""
viz_global_components.py
========================
Visualizadores de componentes globales del modelo PSBP-FD.

Funciones públicas
------------------
plot_global_components   : Panel completo para parámetros globales escalares
                           (muout, N1out, beta0hout, tauhout, alphahout, pijout).
                           Incluye trazas, ACF y distribución posterior.
plot_active_clusters     : Evolución de la cantidad de clusters activos (N1out)
                           a lo largo de las iteraciones, con distribución posterior.

Convención de trazas clave
--------------------------
muout      : (nsim,)     — media global
N1out      : (nsim,)     — clusters activos
beta0hout  : (nsim, N)   — intercepto por átomo
tauhout    : (nsim, N)   — precisión por átomo
alphahout  : (nsim, N-1) — umbral PSBP
pijout     : (nsim, p)   — probabilidad de inclusión (agregada sobre p)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Reutilizar diagnósticos (autocontenidos)
# ─────────────────────────────────────────────────────────────────────────────

def _autocorr(x: np.ndarray, max_lag: int = 80) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    x_c = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x_c, n=nfft)
    acf_full = np.fft.irfft(f * np.conj(f), n=nfft)[:n]
    acf_full /= acf_full[0]
    return acf_full[: max_lag + 1]


def _ess_geyer(x: np.ndarray) -> float:
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


def _aggregate_trace(trace: np.ndarray, burn: int) -> np.ndarray:
    """Promedia sobre dimensiones de átomos/variables y aplica burn-in."""
    trace = np.asarray(trace)
    if trace.ndim == 1:
        agg = trace
    elif trace.ndim == 2:
        agg = trace.mean(axis=1)
    elif trace.ndim == 3:
        agg = trace.mean(axis=(1, 2))
    else:
        raise ValueError(f"ndim={trace.ndim} no soportado")
    return agg[burn:].astype(np.float64)


_CHAIN_COLORS = ["#1a6faf", "#e07b39", "#3aaa35", "#9b59b6", "#c0392b",
                 "#16a085", "#d35400", "#2c3e50"]

# Parámetros globales por defecto con etiquetas LaTeX
_GLOBAL_PARAMS = {
    "muout":     r"$\mu$ (media global)",
    "N1out":     r"$N^{(1)}$ (clusters activos)",
    "beta0hout": r"$\bar{\beta}_{0,h}$ (intercepto agr.)",
    "tauhout":   r"$\bar{\tau}_h$ (precisión agr.)",
    "alphahout": r"$\bar{\alpha}_h$ (umbral PSBP agr.)",
    "pijout":    r"$\bar{p}_{ij}$ (incl. prob. agr.)",
}


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_components(
    models_chains: Dict,
    k: int,
    burn: int,
    n_chains: int,
    params_to_show: Optional[Dict[str, str]] = None,
    max_lag: int = 80,
    figsize_per_param: Tuple[float, float] = (15, 2.5),
    title_prefix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Panel de diagnóstico para parámetros globales del componente k.

    Cada fila muestra: traza | ACF | distribución posterior.
    Incluye ESS y media posterior en el encabezado de cada distribución.

    Parámetros
    ----------
    models_chains    : dict {k: {c: modelo}}
    k                : índice de componente (0-based)
    burn             : iteraciones de burn-in
    n_chains         : número de cadenas
    params_to_show   : dict {trace_key: label_latex}; si None usa los 6 predefinidos
    max_lag          : lags máximos para ACF
    figsize_per_param: (ancho, alto) por fila
    title_prefix     : prefijo para el título principal
    save_path        : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    params = params_to_show or _GLOBAL_PARAMS

    # Filtrar claves que existen en las trazas
    available = list(models_chains[k][0].traces.keys())
    params = {pk: lbl for pk, lbl in params.items() if pk in available}

    n_params = len(params)
    if n_params == 0:
        raise ValueError("Ningún parámetro solicitado existe en traces.")

    fig, axes = plt.subplots(
        n_params, 3,
        figsize=(figsize_per_param[0], figsize_per_param[1] * n_params),
        squeeze=False,
    )
    prefix = f"{title_prefix}  " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Componentes globales — Componente {k + 1}",
        fontsize=12, y=1.002,
    )

    for row, (pkey, plabel) in enumerate(params.items()):
        ax_tr, ax_acf, ax_dist = axes[row]

        chains_post = []
        for c in range(n_chains):
            tr  = models_chains[k][c].traces[pkey]
            agg = _aggregate_trace(tr, burn)
            chains_post.append(agg)

        n_post = len(chains_post[0])
        ci     = 1.96 / np.sqrt(n_post)

        # ── Trazas ──────────────────────────────────────────────────────
        for c, ch in enumerate(chains_post):
            ax_tr.plot(ch, lw=0.6, alpha=0.75,
                       color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)],
                       label=f"cad {c + 1}")
        ax_tr.set_title(f"{plabel}  — trazas", fontsize=8)
        ax_tr.set_xlabel("iter post burn-in", fontsize=7)
        ax_tr.legend(fontsize=6, ncol=n_chains)
        ax_tr.grid(True, alpha=0.3)

        # ── ACF ─────────────────────────────────────────────────────────
        for c, ch in enumerate(chains_post):
            rho = _autocorr(ch, max_lag=max_lag)
            ax_acf.plot(np.arange(max_lag + 1), rho, lw=0.8, alpha=0.8,
                        color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)])
        ax_acf.axhline(0,   color="k", lw=0.5)
        ax_acf.axhline( ci, color="k", ls=":", lw=0.7, label="±1.96/√n")
        ax_acf.axhline(-ci, color="k", ls=":", lw=0.7)
        ax_acf.set_title(f"{plabel}  — ACF", fontsize=8)
        ax_acf.set_xlabel("lag", fontsize=7)
        ax_acf.set_ylim(-0.35, 1.05)
        ax_acf.legend(fontsize=6, loc="upper right")
        ax_acf.grid(True, alpha=0.3)

        # ── Distribución posterior combinada ────────────────────────────
        combined = np.concatenate(chains_post)
        ess      = float(np.mean([_ess_geyer(ch) for ch in chains_post]))
        color_hist = "steelblue" if "beta" in pkey or "mu" in pkey else (
            "darkorange" if "pij" in pkey else "mediumpurple"
        )
        ax_dist.hist(combined, bins=40, color=color_hist, alpha=0.75,
                     edgecolor="white")
        ax_dist.axvline(combined.mean(), color="k", lw=1.2, ls="--",
                        label=f"media={combined.mean():.3f}")
        ax_dist.axvline(np.median(combined), color="crimson", lw=1.0, ls=":",
                        label=f"mediana={np.median(combined):.3f}")
        ax_dist.set_title(
            f"{plabel}  — posterior  ESS≈{ess:.0f}",
            fontsize=8,
        )
        ax_dist.legend(fontsize=6)
        ax_dist.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_active_clusters(
    models_chains: Dict,
    k: int,
    burn: int,
    n_chains: int,
    figsize: Tuple[float, float] = (13, 5),
    title_prefix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Evolución e histograma del número de clusters activos N1out.

    Panel izquierdo : trazas de N1 por cadena.
    Panel derecho   : distribución posterior de N1 (barras discretas).

    Parámetros
    ----------
    models_chains : dict {k: {c: modelo}}
    k             : índice de componente (0-based)
    burn          : iteraciones de burn-in
    n_chains      : número de cadenas
    figsize       : tamaño total de la figura
    title_prefix  : prefijo para el título
    save_path     : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    fig, (ax_tr, ax_dist) = plt.subplots(1, 2, figsize=figsize)
    prefix = f"{title_prefix}  " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Clusters activos N₁ — Componente {k + 1}",
        fontsize=12,
    )

    all_chains = []
    for c in range(n_chains):
        tr  = models_chains[k][c].traces["N1out"]
        agg = _aggregate_trace(tr, burn)
        all_chains.append(agg)
        ax_tr.plot(agg, lw=0.7, alpha=0.8,
                   color=_CHAIN_COLORS[c % len(_CHAIN_COLORS)],
                   label=f"cad {c + 1}")

    ax_tr.set_title(r"Evolución de $N_1$ (clusters activos)", fontsize=10)
    ax_tr.set_xlabel("iteración post burn-in", fontsize=9)
    ax_tr.set_ylabel(r"$N_1$", fontsize=9)
    ax_tr.legend(fontsize=8, ncol=n_chains)
    ax_tr.grid(True, alpha=0.3)

    # Distribución posterior discreta
    combined = np.concatenate(all_chains)
    values, counts = np.unique(combined.round().astype(int), return_counts=True)
    probs  = counts / counts.sum()
    mode   = int(values[np.argmax(probs)])
    mean_v = float(combined.mean())

    ax_dist.bar(values, probs * 100, color="steelblue", alpha=0.8,
                edgecolor="white", label="Distribución posterior")
    ax_dist.axvline(mean_v, color="crimson", lw=1.5, ls="--",
                    label=f"media={mean_v:.2f}")
    ax_dist.axvline(mode, color="k", lw=1.5, ls=":",
                    label=f"moda={mode}")
    ax_dist.set_title(r"Distribución posterior de $N_1$", fontsize=10)
    ax_dist.set_xlabel(r"$N_1$", fontsize=9)
    ax_dist.set_ylabel("Probabilidad posterior (%)", fontsize=9)
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
