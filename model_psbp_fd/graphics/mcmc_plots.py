"""
mcmc_plots.py
-------------
Trace plots y densidades marginales para diagnóstico MCMC.
"""
from __future__ import annotations
import numpy as np


def trace_plot(chain: np.ndarray, param_name: str = "", ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(chain, linewidth=0.6, color="steelblue")
    ax.set_title(f"Trace: {param_name}")
    ax.set_xlabel("Iteración")
    return ax


def marginal_density(chain: np.ndarray, param_name: str = "", ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(chain, bins=40, density=True, color="steelblue", alpha=0.7)
    ax.set_title(f"Densidad marginal: {param_name}")
    return ax
