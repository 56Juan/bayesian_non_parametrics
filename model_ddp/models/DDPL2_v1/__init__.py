"""
DDPL2 v1

Modelo de Procesos dirichlet dependientes con dos priors Dirichlet.
Implementación híbrida Python / C++ (pybind11).
"""

from .DDPL2_v1 import DDPLinearSpline2
from . import ddp2_cpp

__all__ = [
    "DDPLinearSpline2",
    "ddp2_cpp",
]
