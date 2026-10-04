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
                                   retencion temporal. Es la FPCA estatica
                                   del proyecto.
    functions_odpc.py              Componentes dinamicas UNILATERALES (ODPC;
                                   Pena, Smucler & Yohai 2019) sobre los scores
                                   de la estatica; corridas 110-112.
    functions_standarize.py        Estandarizacion de scores con persistencia.
"""

from .functions_standarize import DataStandardizer
from .functions_repre_functional import FunctionalRepresentation
from .functions_fpca import FPCA_L2, base_en_grilla
from .functions_odpc import ODPC_Funcional

__all__ = [
    "DataStandardizer",
    "FunctionalRepresentation",
    "FPCA_L2",
    "ODPC_Funcional",
    "base_en_grilla",
]
