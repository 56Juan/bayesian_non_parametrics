"""
simulation_pipeline.py
----------------------
Pipeline completo: genera DGP → estandariza → ajusta modelo → evalúa.
"""
from __future__ import annotations
import numpy as np
from model_psbp_fd.models import PSBPFunctional
from model_psbp_fd.data_functions.preprocessors import standardize, build_lags
from model_psbp_fd.fit.metrics import rmse_per_component


class SimulationPipeline:
    def __init__(self, model_config: dict, seed: int = 0) -> None:
        self.model_config = model_config
        self.seed = seed
        self.model_: PSBPFunctional | None = None
        self.results_: dict = {}

    def run(self, THETA: np.ndarray, nsim: int = 3000, burn: int = 500) -> dict:
        model = PSBPFunctional(**self.model_config, seed=self.seed)
        model.fit(THETA, nsim=nsim, burn=burn)
        self.model_ = model

        chains    = model.chains_
        inE_mean  = chains["inE"].mean(axis=0)   # n x K (escala estd.)
        inE_orig  = inE_mean * chains["theta_std"] + chains["theta_mean"]
        THETA_s, *_ = standardize(THETA)
        _, Theta_resp, _, _ = (None, THETA[model.c:], None, None)

        self.results_ = {
            "rmse":    rmse_per_component(inE_orig, Theta_resp),
            "N1_mean": chains["N1"].mean(),
            "chains":  chains,
        }
        return self.results_
