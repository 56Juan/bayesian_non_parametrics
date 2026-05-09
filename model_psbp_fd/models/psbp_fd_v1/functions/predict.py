"""
predict.py
----------
Cálculo de la media predictiva posterior para nuevas observaciones.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def posterior_predictive_mean(
    chains: dict[str, np.ndarray],
    Theta_lag_new: np.ndarray,
    Theta_ext_new: np.ndarray,
    L: int,
) -> np.ndarray:
    """
    Calcula E[theta_{t+1} | Theta_c, datos] promediando sobre las cadenas post-burn.

    Parameters
    ----------
    chains        : dict devuelto por GibbsSampler.run()
    Theta_lag_new : (n_new, c*K)     rezagos estandarizados
    Theta_ext_new : (n_new, c*K+1)   rezagos con intercepto estandarizado
    L             : nivel de truncación

    Returns
    -------
    pred_mean : (n_new, K) en escala estandarizada
    """
    nsim_post = chains["alphah"].shape[0]
    n_new, K  = Theta_ext_new.shape[0], chains["Bl"].shape[1]

    pred_accum = np.zeros((n_new, K))

    for gt in range(nsim_post):
        alphah = chains["alphah"][gt]          # (L-1,)
        psih   = chains["psih"][gt]            # (L-1, p)
        Gammah = chains["Gammah"][gt]          # (L-1, p)
        Bl     = chains["Bl"][gt]              # (K, d, L)

        for i in range(n_new):
            xi  = Theta_lag_new[i]
            vhx = np.array([
                norm.cdf(alphah[h] - np.sum(psih[h] * np.abs(xi - Gammah[h])))
                for h in range(L - 1)
            ])
            phx    = np.empty(L)
            phx[0] = vhx[0]
            for h in range(1, L - 1):
                phx[h] = vhx[h] * np.prod(1 - vhx[:h])
            phx[L - 1] = np.prod(1 - vhx)

            xi_ext = Theta_ext_new[i]
            inE_i  = sum(phx[l] * (Bl[:, :, l] @ xi_ext) for l in range(L))
            pred_accum[i] += inE_i

    return pred_accum / nsim_post


def posterior_predictive_samples(
    chains: dict[str, np.ndarray],
    Theta_lag_new: np.ndarray,
    Theta_ext_new: np.ndarray,
    L: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera muestras de la distribución predictiva (para bandas credibles).

    Returns
    -------
    samples : (nsim_post, n_new, K)
    """
    # TODO: implementar muestreo completo de la predictiva
    raise NotImplementedError("Pendiente de implementación.")
