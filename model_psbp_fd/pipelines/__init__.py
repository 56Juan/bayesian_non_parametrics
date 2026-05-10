# VERSION 2 - Actualizado para el nuevo FARSimulator

from .simulation_pipeline_linear import (
    FunctionalDomain,
    FARSimulator,
    gaussian_integral_kernel,
    build_integral_matrix,
)

__all__ = [
    "FunctionalDomain",
    "FARSimulator",
    "gaussian_integral_kernel",
    "build_integral_matrix",
]