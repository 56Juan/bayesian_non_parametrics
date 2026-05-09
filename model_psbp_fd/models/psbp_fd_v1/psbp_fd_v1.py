"""
psbp_fd_v1.py
-------------
Clase pública del modelo PSBP para series de tiempo funcionales (FAR-c).

Uso básico
----------
>>> from model_psbp_fd.models import PSBPFunctional
>>> model = PSBPFunctional(K=4, c=2, L=20, M=50)
>>> model.fit(THETA, nsim=3000, burn=500)
>>> pred = model.predict(THETA_new)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from .functions.sampler import GibbsSampler
from .functions.predict import posterior_predictive_mean


@dataclass
class PSBPFunctional:
    """
    PSBP extendido a datos funcionales (Chung & Dunson, 2009).

    Parameters
    ----------
    K    : número de funciones de base
    c    : orden autorregresivo
    L    : nivel de truncación del proceso stick-breaking
    M    : puntos de grilla por dimensión del predictor de pesos
    seed : semilla aleatoria (reproducibilidad)
    """
    K:    int   = 4
    c:    int   = 1
    L:    int   = 20
    M:    int   = 50
    seed: int | None = None

    # ── estado posterior (se pobla al llamar fit()) ─────────────────────
    chains_:      dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    hyperparams_: dict[str, Any]        = field(default_factory=dict, repr=False)
    is_fitted_:   bool                  = False

    def fit(
        self,
        THETA: np.ndarray,
        nsim: int = 3000,
        burn: int = 500,
        hyperparams: dict | None = None,
    ) -> "PSBPFunctional":
        """
        Ajusta el modelo mediante Gibbs sampler.

        Parameters
        ----------
        THETA      : coeficientes funcionales (T, K)
        nsim       : número total de iteraciones MCMC
        burn       : iteraciones de calentamiento a descartar
        hyperparams: dict con hiperparámetros (sobreescribe defaults del config)

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.seed)
        sampler = GibbsSampler(
            K=self.K, c=self.c, L=self.L, M=self.M,
            hyperparams=hyperparams or {},
            rng=rng,
        )
        self.chains_      = sampler.run(THETA, nsim=nsim, burn=burn)
        self.hyperparams_ = sampler.hyperparams_
        self.is_fitted_   = True
        return self

    def predict(
        self,
        Theta_lag_new: np.ndarray,
        Theta_ext_new: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula la media predictiva posterior para nuevas observaciones.

        Parameters
        ----------
        Theta_lag_new : rezagos estandarizados (n_new, c*K)
        Theta_ext_new : rezagos con intercepto (n_new, c*K+1)

        Returns
        -------
        pred_mean : media predictiva (n_new, K)
        """
        if not self.is_fitted_:
            raise RuntimeError("Llamar a fit() antes de predict().")
        return posterior_predictive_mean(
            chains=self.chains_,
            Theta_lag_new=Theta_lag_new,
            Theta_ext_new=Theta_ext_new,
            L=self.L,
        )
