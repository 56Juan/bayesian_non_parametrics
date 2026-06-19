"""
viz_prediction.py
=================
Visualizadores de evaluación predictiva para el modelo PSBP-FD.

Funciones públicas
------------------
plot_scatter_theta        : Scatter plot θ_observado vs θ_predicho por componente.
plot_functional_comparison: Comparativa de reconstrucción funcional:
                            X_verdadera | proyección B-spline | predicción PSBP.

Contexto
--------
Después del ajuste PSBP-FD, se obtienen:
    eval_results[k]["y_obs"]  : (T_eff,) scores observados del componente k
    eval_results[k]["y_hat"]  : (T_eff,) scores predichos del componente k
    eval_results[k]["incl_named"] : pd.Series con P(γ=1|data) por variable

La comparativa funcional reconstruye curvas en la escala de X_true usando
la instancia FunctionalRepresentation `fr` ajustada.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Paletas
# ─────────────────────────────────────────────────────────────────────────────
_COMP_COLORS = ["#1a6faf", "#e07b39", "#3aaa35", "#9b59b6", "#c0392b",
                "#16a085", "#d35400", "#2c3e50"]


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_theta(
    eval_results: Dict,
    n_components: int,
    # [FIX] Las cantidades graficadas son scores FPCA (ξ), no coeficientes
    # B-spline (θ). Se corrige el etiquetado por defecto y se parametriza
    # el símbolo para usos futuros.
    title: str = r"Ajuste por componente (in-sample): $\xi_{t,k}$ observado vs predicho",
    symbol: str = r"\xi",
    n_cols: int = 4,
    figsize_per_panel: Tuple[float, float] = (5.0, 4.5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot θ_observado vs θ_predicho para cada componente k.

    Incluye:
    - Línea identidad (ideal).
    - Regresión OLS superpuesta.
    - Métricas RMSE, R² y correlación en el título de cada panel.
    - Top-3 variables con mayor P(γ=1|data).

    Parámetros
    ----------
    eval_results      : dict {k: {"y_obs": arr, "y_hat": arr, "incl_named": Series}}
    n_components      : número de componentes K
    title             : título global de la figura
    n_cols            : número de columnas en la grilla de paneles
    figsize_per_panel : (ancho, alto) por panel
    save_path         : ruta de guardado opcional

    Retorna
    -------
    fig : Figure
    """
    n_cols_eff = min(n_components, n_cols)
    n_rows     = (n_components + n_cols_eff - 1) // n_cols_eff

    fig, axes = plt.subplots(
        n_rows, n_cols_eff,
        figsize=(figsize_per_panel[0] * n_cols_eff, figsize_per_panel[1] * n_rows),
    )
    axes = np.atleast_1d(axes).flatten()

    for k in range(n_components):
        ax     = axes[k]
        y_obs  = eval_results[k]["y_obs"]
        y_hat  = eval_results[k]["y_hat"]

        ss_res = np.sum((y_obs - y_hat) ** 2)
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        rmse   = np.sqrt(ss_res / len(y_obs))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        corr   = np.corrcoef(y_obs, y_hat)[0, 1]

        ax.scatter(y_obs, y_hat, s=14, alpha=0.55,
                   color=_COMP_COLORS[k % len(_COMP_COLORS)],
                   edgecolors="white", linewidths=0.3)

        # Línea identidad
        lims = [min(y_obs.min(), y_hat.min()), max(y_obs.max(), y_hat.max())]
        pad  = 0.05 * (lims[1] - lims[0])
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, "k--", lw=1.0, label="identidad (ideal)")

        # Regresión OLS
        if _HAS_SCIPY:
            slope, intercept, *_ = _stats.linregress(y_obs, y_hat)
            xs = np.array(lims)
            ax.plot(xs, intercept + slope * xs, "r-", lw=1.2, alpha=0.7,
                    label=f"OLS: y={slope:.2f}·x{intercept:+.2f}")

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_box_aspect(1)

        # Top-3 inclusiones
        incl = eval_results[k].get("incl_named", None)
        if incl is not None:
            top3 = incl.sort_values(ascending=False).head(3)
            incl_str = " | ".join(
                f"{name.replace('theta_', 'θ').replace('_lag', '_t-')}: {val:.2f}"
                for name, val in top3.items()
            )
        else:
            incl_str = ""

        ax.set_xlabel(rf"${symbol}_{{{k+1}}}$ observado", fontsize=9)
        ax.set_ylabel(rf"$\hat{{{symbol}}}_{{{k+1}}}$ predicho", fontsize=9)
        ax.set_title(
            rf"Comp {k+1}: RMSE={rmse:.4f}, $R^2$={r2:.3f}, corr={corr:.3f}"
            + (f"\nTop incl: {incl_str}" if incl_str else ""),
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.4)

    # Ocultar paneles sobrantes
    for ax in axes[n_components:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_functional_comparison(
    eval_results: Dict,
    fr,
    domain_grid: np.ndarray,
    X_true: np.ndarray,
    n_components: int,
    n_lags: int = 1,
    n_snapshots: int = 5,
    t_indices: Optional[List[int]] = None,
    title: str = (r"Reconstrucción funcional: "
                  r"$\tilde X_t$ verdadera vs representación vs predicción PSBP"),
    # [FIX] La capa intermedia depende del operador `fr` que se pase
    # (B-spline real o proxy FPCA); la etiqueta debe declararlo.
    repr_label: str = "representación funcional",
    figsize_per_col: Tuple[float, float] = (3.5, 4.0),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, dict]:
    """
    Comparativa visual de la reconstrucción funcional.

    Para cada snapshot t muestra tres curvas superpuestas:
        (negro) X_t verdadera (en la escala de X_true).
        (azul)  Proyección B-spline de X_t (representación funcional).
        (rojo)  Predicción PSBP reconstruida.

    Parámetros
    ----------
    eval_results  : dict {k: {"y_obs": arr, "y_hat": arr}}
    fr            : FunctionalRepresentation ajustada (post fit_transform)
    domain_grid   : (G,) grilla del dominio S
    X_true        : (T, G) curvas verdaderas (completas, incluyendo lags)
    n_components  : número de componentes K
    n_lags        : número de lags usados en el modelo (para alinear índices)
    n_snapshots   : número de instantes temporales a mostrar
    t_indices     : lista de índices t específicos (0-based dentro de T_eff);
                    si None, se distribuyen uniformemente
    title         : título global
    figsize_per_col: (ancho, alto) por columna (snapshot)
    save_path     : ruta de guardado opcional

    Retorna
    -------
    fig     : Figure
    metrics : dict con RMSE_repr, RMSE_model, RMSE_total (escala de X_true)
    """
    # ── Construir matrices de scores ─────────────────────────────────────
    THETA_obs  = np.column_stack([eval_results[k]["y_obs"] for k in range(n_components)])
    THETA_pred = np.column_stack([eval_results[k]["y_hat"] for k in range(n_components)])

    # ── Reconstrucciones funcionales ─────────────────────────────────────
    X_repr_obs  = fr.reconstruct(THETA_obs)    # (T_eff, G)
    X_repr_pred = fr.reconstruct(THETA_pred)   # (T_eff, G)
    X_true_eff  = X_true[n_lags:, :]           # (T_eff, G)

    T_eff = X_true_eff.shape[0]

    # ── Métricas de descomposición ────────────────────────────────────────
    err_repr  = X_true_eff - X_repr_obs
    err_model = X_repr_obs - X_repr_pred
    err_total = X_true_eff - X_repr_pred
    sd        = X_true_eff.std()

    rmse_repr  = float(np.sqrt(np.mean(err_repr  ** 2)))
    rmse_model = float(np.sqrt(np.mean(err_model ** 2)))
    rmse_total = float(np.sqrt(np.mean(err_total ** 2)))

    metrics = {
        "rmse_repr":  rmse_repr,
        "rmse_model": rmse_model,
        "rmse_total": rmse_total,
        "nrmse_repr":  rmse_repr  / sd if sd > 0 else np.nan,
        "nrmse_model": rmse_model / sd if sd > 0 else np.nan,
        "nrmse_total": rmse_total / sd if sd > 0 else np.nan,
    }

    print("── Métricas de reconstrucción (escala de X_true) ──")
    print(f"  RMSE representación : {rmse_repr:.6f}   NRMSE={metrics['nrmse_repr']:.3f}")
    print(f"  RMSE modelo         : {rmse_model:.6f}   NRMSE={metrics['nrmse_model']:.3f}")
    print(f"  RMSE total          : {rmse_total:.6f}   NRMSE={metrics['nrmse_total']:.3f}")
    if rmse_total > 0:
        print(f"  Ratio modelo/total  : {rmse_model / rmse_total:.3f}")

    # ── Selección de snapshots ────────────────────────────────────────────
    if t_indices is None:
        t_indices = list(np.linspace(0, T_eff - 1, n_snapshots, dtype=int))

    n_show = len(t_indices)
    fig, axes = plt.subplots(
        1, n_show,
        figsize=(figsize_per_col[0] * n_show, figsize_per_col[1]),
        sharey=True,
    )
    if n_show == 1:
        axes = [axes]

    for col_idx, t_idx in enumerate(t_indices):
        ax     = axes[col_idx]
        t_real = t_idx + n_lags + 1
        rmse_snap = float(np.sqrt(np.mean((X_true_eff[t_idx] - X_repr_pred[t_idx]) ** 2)))

        ax.plot(domain_grid, X_true_eff[t_idx],
                color="black", lw=1.4, alpha=0.85,
                label=r"$\tilde X_t$ verdadera" if col_idx == 0 else None)
        ax.plot(domain_grid, X_repr_obs[t_idx],
                color="steelblue", lw=1.4, ls=":",
                label=repr_label if col_idx == 0 else None)
        ax.plot(domain_grid, X_repr_pred[t_idx],
                color="crimson", lw=1.4, ls="--",
                label="predicción PSBP" if col_idx == 0 else None)

        ax.set_title(rf"$t = {t_real}$" + f"\nRMSE={rmse_snap:.3f}", fontsize=9)
        ax.set_xlabel("$s$", fontsize=9)
        ax.grid(True, alpha=0.35)

    axes[0].set_ylabel(r"$X_t(s)$", fontsize=10)
    axes[0].legend(fontsize=8, loc="best")

    fig.suptitle(title, fontsize=11, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, metrics
