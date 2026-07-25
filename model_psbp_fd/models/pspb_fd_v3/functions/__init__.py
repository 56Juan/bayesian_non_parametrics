"""
Subpaquete functions — Componentes de la inferencia predictiva de PSBP-FD v3.

    predict.py      Distribucion predictiva de un PSBPM univariado a partir de
                    sus trazas: momentos con descomposicion de varianza total,
                    muestreo de la predictiva, densidad cerrada e intervalos.
    propagation.py  Transporte de la predictiva de los scores a la predictiva
                    funcional: agrupacion de cadenas, des-estandarizacion,
                    reconstruccion FPCA e incorporacion del residuo de
                    representacion.

Esta version NO contiene muestreador: el ajuste ocurre en MATLAB
(`psbp_train.m`) y aqui solo se consume el resultado.
"""

from .predict import PSBPPredictor, pesos_probit, medias_componente
from .propagation import (
    PropagadorFuncional,
    agrupar_muestras_cadenas,
    bandas_puntuales,
    muestrear_scores,
    residuos_representacion,
)

__all__ = [
    "PSBPPredictor",
    "pesos_probit",
    "medias_componente",
    "PropagadorFuncional",
    "agrupar_muestras_cadenas",
    "bandas_puntuales",
    "muestrear_scores",
    "residuos_representacion",
]
