"""
PSBP Normal v2

Modelo PROBIT Stick-Breaking Process con kernel normal.
Implementación híbrida Python / C++ (pybind11).
"""

from .PSBP_normal_v2 import PSBPNormal
from . import psbp_cpp

__all__ = [
    "PSBPNormal",
    "psbp_cpp",
]
