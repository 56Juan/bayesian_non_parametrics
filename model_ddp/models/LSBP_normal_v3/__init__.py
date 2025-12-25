"""
LSBP Normal v3

Modelo Logistic Stick-Breaking Process con kernel normal.
Implementación híbrida Python / C++ (pybind11).
"""

from .LSBP_normal_v3 import LSBPNormal
from . import lsbp_cpp

__all__ = [
    "LSBPNormal",
    "lsbp_cpp",
]
