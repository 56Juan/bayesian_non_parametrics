"""
mv_psbp — PSBPM-FD conjunto (multivariado) sobre los coeficientes de la
representacion funcional. Vive en notebooks/experimento/ y usa el paquete
model_psbp_fd sin modificarlo.

Modulos:
  rutas        directorios del experimento (todo bajo notebooks/experimento/)
  preparacion  representacion, blanqueo por la Gram, datasets y contrato MATLAB
  trazas       lectura de las trazas conjuntas y predictor (pesos, media, muestras)
  evaluacion   prediccion a h=1, bandas, ventana movil (las diez metricas)
  comparacion  FAR(p) con kn por hold-out, PSBPM-FD univariado de los reportes
               externos, tablas 96-99 y figuras 76/77
  convergencia diagnosticos sobre cantidades invariantes a la permutacion
"""
from .rutas import RAIZ_EXPERIMENTO, construir_paths_mv, experiment_id_mv
