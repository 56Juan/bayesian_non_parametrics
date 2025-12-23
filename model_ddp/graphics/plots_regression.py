# plots_regression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plot_density(y_true, y_pred, ax, title=None, xlabel="Y"):
    ax.hist(y_true, bins=30, alpha=0.5, density=True, label="Real")
    ax.hist(y_pred, bins=30, alpha=0.5, density=True, label="Predicho")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densidad")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)


def plot_scatter(y_true, y_pred, ax, title=None):
    r2 = r2_score(y_true, y_pred)
    ax.scatter(y_true, y_pred, alpha=0.5, s=12)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    ax.plot(lims, lims, "r--", lw=2, label="Perfecta")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    if title:
        ax.set_title(f"{title} (R²={r2:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)


def plot_residuals(y_true, y_pred, ax, title=None):
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=12)
    ax.axhline(0.0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Residuo")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_regression_analysis(
    splits,
    output_path,
    model_name="Modelo"
):
    """
    splits: lista de tuplas [(y_true, y_pred, label), ...]
    """

    # =========================
    # Densidades + Scatter
    # =========================
    fig, axes = plt.subplots(2, len(splits), figsize=(6 * len(splits), 10))
    fig.suptitle(f"Análisis de Regresión - {model_name}", fontsize=14, fontweight="bold")

    for i, (y_true, y_pred, label) in enumerate(splits):
        plot_density(
            y_true, y_pred,
            ax=axes[0, i],
            title=f"Densidad - {label}"
        )
        plot_scatter(
            y_true, y_pred,
            ax=axes[1, i],
            title=label
        )

    plt.tight_layout()
    plt.savefig(f"{output_path}/analisis_regresion.png", dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # Residuos
    # =========================
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5))
    fig.suptitle(f"Residuos - {model_name}", fontsize=14, fontweight="bold")

    for i, (y_true, y_pred, label) in enumerate(splits):
        plot_residuals(
            y_true, y_pred,
            ax=axes[i],
            title=f"Residuos - {label}"
        )

    plt.tight_layout()
    plt.savefig(f"{output_path}/residuos_regresion.png", dpi=300, bbox_inches="tight")
    plt.close()
