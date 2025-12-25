"""
LSBP Laplace v1

Modelo Logistic Stick-Breaking Process con kernel laplace.
Implementación híbrida Python / C++ (pybind11).
"""

from .LSBP_laplace_v1 import LSBPLaplace
from . import lsbp_laplace_cpp

__all__ = [
    "LSBPLaplace",
    "lsbp_laplace_cpp",
]
