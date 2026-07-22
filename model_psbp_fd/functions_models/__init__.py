"""
functions_models
================
Preprocesamiento de la representacion funcional.

    functions_repre_functional.py  Suavizado en bases (B-spline, Fourier) con
                                   proyeccion L2 o discreta y cuadratura
                                   trapezoidal, la regla comun del proyecto.
                                   Es exclusivamente un suavizador.
    functions_fpca.py              FPCA generalizado en metrica L2 sobre los
                                   coeficientes de una base no ortonormal
                                   (problema propio C u = lambda W u), con
                                   patron fit/transform para el esquema de
                                   retencion temporal. Es la UNICA FPCA del
                                   proyecto.
    functions_standarize.py        Estandarizacion de scores con persistencia.
"""

from .functions_standarize import DataStandardizer
from .functions_repre_functional import FunctionalRepresentation
from .functions_fpca import FPCA_L2, base_en_grilla

__all__ = [
    "DataStandardizer",
    "FunctionalRepresentation",
    "FPCA_L2",
    "base_en_grilla",
]
