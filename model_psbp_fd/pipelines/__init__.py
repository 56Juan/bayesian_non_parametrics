# VERSION 3 - Añadido el esquema de simulación no lineal FAR(p)

from .simulation_pipeline_linear import (
    FunctionalDomain,
    FARSimulator,
    gaussian_integral_kernel,
    build_integral_matrix,
)

from .simulation_pipeline_nonlinear import (
    build_lag_operators,
    companion_spectral_radius,
    NonlinearFARMap,
    ThresholdFAR,
    PointwiseNonlinearFAR,
    BilinearFAR,
    MixtureStateFAR,
    NonlinearFARSimulator,
)

__all__ = [
    # --- Esquema lineal FAR(1) ---
    "FunctionalDomain",
    "FARSimulator",
    "gaussian_integral_kernel",
    "build_integral_matrix",
    # --- Esquema no lineal FAR(p) ---
    "build_lag_operators",
    "companion_spectral_radius",
    "NonlinearFARMap",
    "ThresholdFAR",
    "PointwiseNonlinearFAR",
    "BilinearFAR",
    "MixtureStateFAR",
    "NonlinearFARSimulator",
]
