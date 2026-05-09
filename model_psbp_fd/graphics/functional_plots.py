"""
functional_plots.py
-------------------
Visualización de curvas funcionales y bandas credibles.
"""
from __future__ import annotations
import numpy as np


def plot_functional_bands(
    grid: np.ndarray,
    pred_samples: np.ndarray,
    true_curve: np.ndarray | None = None,
    alpha: float = 0.95,
    ax=None,
):
    """
    Grafica la media posterior y banda credible (alpha*100)% para una curva funcional.

    Parameters
    ----------
    grid         : puntos de evaluación (G,)
    pred_samples : muestras predictivas (nsim, G)
    true_curve   : curva verdadera (G,), opcional
    alpha        : cobertura de la banda
    ax           : matplotlib Axes
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    lo = (1 - alpha) / 2 * 100
    hi = 100 - lo
    mean_ = pred_samples.mean(axis=0)
    lo_   = np.percentile(pred_samples, lo, axis=0)
    hi_   = np.percentile(pred_samples, hi, axis=0)

    ax.plot(grid, mean_, "k-",  linewidth=1.5, label="Media posterior")
    ax.fill_between(grid, lo_, hi_, alpha=0.25, color="steelblue",
                    label=f"Banda {int(alpha*100)}%")
    if true_curve is not None:
        ax.plot(grid, true_curve, "r--", linewidth=1.5, label="Verdadera")
    ax.legend()
    return ax
