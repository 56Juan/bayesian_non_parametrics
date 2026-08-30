"""
utils
=====
Utilidades transversales del proyecto.

    quadrature.py  Cuadratura L2 sobre la grilla del dominio (regla trapezoidal,
                   unica del proyecto): pesos, integracion, producto interno,
                   matriz de Gram y norma L2.
    linalg.py      Algebra lineal numericamente estable: simetrizacion y
                   factorizacion de Cholesky con jitter incremental RELATIVO a
                   la escala de la matriz, con fallback espectral opcional.
                   Es la unica factorizacion del proyecto: `pipelines.sim_comun`
                   la re-exporta bajo el nombre de dominio `factor_cholesky`.
    raiz.py        Localizacion de la raiz del proyecto.
    trazas.py      Normalizacion de forma de las trazas `.mat`: MATLAB descarta
                   las dimensiones singleton finales, de modo que con una sola
                   covariable (p = 1, el punto M = 1 del barrido) las trazas de
                   tres ejes llegan con dos. Definicion unica de esa correccion.
    progreso.py    Salida de progreso por consola para los bucles largos de
                   `fit` y `graphics`. Definicion unica del formato de la linea
                   de progreso; los modulos la piden con `verbose=True` y nunca
                   imprimen por su cuenta.
"""

from .quadrature import (
    pesos_trapezoidales,
    integrar,
    producto_interno,
    gram,
    norma_L2,
)
from .linalg import sym, safe_chol, es_triangular_inferior
from .raiz import get_project_root
from .trazas import normalizar_trazas_mat, asegurar_3d
from .progreso import Progreso, aviso

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
    "es_triangular_inferior",
    # Infraestructura
    "get_project_root",
    "normalizar_trazas_mat",
    "asegurar_3d",
    "Progreso",
    "aviso",
]
