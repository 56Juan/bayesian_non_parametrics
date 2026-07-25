"""
psbp_fd_v3
==========
Inferencia predictiva funcional del modelo PSBP-FD.

Diferencia con las versiones anteriores
---------------------------------------
Las versiones 1 y 2 incluian muestreador y predictor en Python. La version 3
abandona el muestreador ---el ajuste se realiza en MATLAB con `psbp_train.m`,
que es la implementacion de referencia del estudio--- y se concentra en lo que
faltaba: construir la distribucion predictiva COMPLETA y transportarla del
espacio de los scores al de las curvas.

Los dos cambios sustantivos son:

1. La desviacion que se reporta en `predict(return_std=True)` es ahora la
   PREDICTIVA, obtenida por la ley de varianza total

       Var[y|x] = E_t[ Var(y | x, theta_t) ] + Var_t[ E(y | x, theta_t) ],

   y no unicamente el segundo termino, que era lo que devolvia la v2. Las
   bandas y la cobertura construidas con el segundo termino solo son
   sistematicamente demasiado angostas.

2. Se incorpora el muestreo de la predictiva. Sin extracciones no pueden
   calcularse las reglas de puntuacion propias del proyecto ---CRPS muestral,
   puntaje de energia, PIT--- y la evaluacion queda reducida a la aproximacion
   gaussiana de dos momentos, que descarta la multimodalidad y la asimetria
   que el modelo existe para capturar.

Contenido
---------
    psbp_fd_v3.py            Orquestador `PSBP_FD_v3` y lector de trazas .mat.
    functions/predict.py     Predictiva de un PSBPM univariado.
    functions/propagation.py Transporte de scores a curvas.
"""

from .psbp_fd_v3 import PSBP_FD_v3, cargar_trazas_mat
from .functions import (
    PSBPPredictor,
    PropagadorFuncional,
    agrupar_muestras_cadenas,
    bandas_puntuales,
    medias_componente,
    muestrear_scores,
    pesos_probit,
    residuos_representacion,
)

__all__ = [
    # Orquestador y lectura de trazas
    "PSBP_FD_v3",
    "cargar_trazas_mat",
    # Predictiva por score
    "PSBPPredictor",
    "pesos_probit",
    "medias_componente",
    # Propagacion funcional
    "PropagadorFuncional",
    "muestrear_scores",
    "agrupar_muestras_cadenas",
    "bandas_puntuales",
    "residuos_representacion",
]
