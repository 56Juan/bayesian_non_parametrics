"""
utils
=====
Utilidades transversales del proyecto.

    quadrature.py  Cuadratura L2 sobre la grilla del dominio (regla trapezoidal,
                   unica del proyecto): pesos, integracion, producto interno,
                   matriz de Gram y norma L2.
    linalg.py      Algebra lineal numericamente estable: simetrizacion y
                   factorizacion de Cholesky con jitter incremental.
    raiz.py        Localizacion de la raiz del proyecto.
"""

from .quadrature import (
    pesos_trapezoidales,
    integrar,
    producto_interno,
    gram,
    norma_L2,
)
from .linalg import sym, safe_chol
from .raiz import get_project_root

__all__ = [
    # Cuadratura L2
    "pesos_trapezoidales",
    "integrar",
    "producto_interno",
    "gram",
    "norma_L2",
    # Algebra lineal estable
    "sym",
    "safe_chol",
    # Infraestructura
    "get_project_root",
]