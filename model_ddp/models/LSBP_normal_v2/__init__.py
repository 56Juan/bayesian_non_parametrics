"""
LSBP Normal v2

Modelo Logistic Stick-Breaking Process con kernel normal.
Implementación híbrida Python / C++ (pybind11).
"""

# Clase pública del modelo (wrapper Python)
from .LSBP_normal_v2 import LSBPNormal

# Backend C++ (opcional, depende de si el módulo está compilado)
try:
    from . import lsbp_cpp
except ImportError:
    lsbp_cpp = None

__all__ = [
    "LSBPNormal",
    "lsbp_cpp",
]
