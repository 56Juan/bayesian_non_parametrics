"""
pipelines
=========
Generacion de datos y contrato de artefactos entre los flujos de trabajo.

1. Generadores de datos. Los escenarios vigentes se simulan sobre los SCORES de
   una base de Fourier conocida, sin ruido de medicion: `sim_scores_comun` es el
   motor comun, `sim_escenario_TAR` el generador de la corrida 200,
   `sim_escenario_GARCH` el de la 201 y `sim_escenario_C1` el de la 202.
   `sim_comun` aloja la configuracion de observacion, la salida y la
   persistencia de un escenario; `sim_escenario_C` la base de Fourier.

2. Contrato de persistencia (`artifacts.py`). Funciones emparejadas de escritura
   y lectura de los artefactos que produce el preprocesamiento y consumen MATLAB
   y la evaluacion, para que la convencion de nombres y las garantias de forma
   tengan una sola definicion. `verificar_contrato` cruza manifest,
   hiperparametros y artefactos FPCA antes de que el analisis comience.

`sim_escenario_1/B1/B2/B3` y `sim_series_clasicas` no tienen corrida viva y se
conservan por dependencias. Los generadores de corridas retiradas viven en
`pipelines/deprecated/`: siguen importables desde ahi, no desde este paquete.
"""

# ══════════════════════════════════════════════════════════════════════════
# 1. GENERADORES DE DATOS
# ══════════════════════════════════════════════════════════════════════════

# --- Componentes transversales ---
from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    grilla_regular,
    media_nula,
    evaluar_media,
    pesos_trapezoidales,
    norma_hilbert_schmidt,
    matriz_operador_ar,
    matriz_covarianza_innovacion,
    factor_cholesky,
    generador_innovacion,
    semillas_replicas,
    aplicar_ruido_observacion,
    diagnostico_comun,
    guardar_escenario,
    cargar_escenario,
)

# --- Escenario 1: FAR(1) lineal gaussiano homogeneo ---
from .sim_escenario_1 import (
    ConfigEscenario1,
    generar_escenario_1,
    resumen_escenario_1,
    simular_trayectoria_far1,
)


# --- Series de tiempo clasicas: mecanismo comun a A-1, A-2 y A-3 ---
from .sim_series_clasicas import (
    acf_empirica_serie,
    bloques_de_serie,
    diagnostico_serie_escalar,
    generar_serie_segmentada,
    oraculo_lineal_un_rezago,
    r2_empirico_media_condicional,
)



# --- Algoritmos B-1, B-2 y B-3: mezcla de mecanismos funcionales (anexo, seccion B) ---
from .sim_escenario_B1 import (
    ConfigEscenarioB1,
    calcular_rasgos,
    funciones_rasgo,
    generar_escenario_B1,
    media_seno,
    medias_mezcla,
    resumen_escenario_B1,
    simular_mezcla_curvas,
    transformar_identidad,
)
from .sim_escenario_B2 import (
    B_MECANISMOS_B2,
    ConfigEscenarioB2,
    generar_escenario_B2,
    resumen_escenario_B2,
    transformar_rasgos_B2,
)
from .sim_escenario_B3 import (
    A_ASIGNACION_B3,
    ConfigEscenarioB3,
    generar_escenario_B3,
    probabilidades_softmax,
    resumen_escenario_B3,
)

# --- Base de Fourier ortonormal de los escenarios sobre scores ---
from .sim_escenario_C import base_fourier, gram_base
# --- Algoritmos C-1, C-2 y C-3 del anexo vigente: mezcla de mecanismos en el
#     espacio de scores (generador de la corrida 202).
from .sim_escenario_C1 import (
    ConfigEscenarioC1,
    MecanismoScores,
    generar_escenario_C1,
    generar_mezcla_scores,
    resumen_escenario_C1,
    resumen_mezcla_scores,
    simular_mezcla_scores,
)
# Escenario sobre scores 201 (GARCH); el TAR es la corrida 200 y la multimodalidad la 202.
from .sim_escenario_GARCH import (
    ConfigEscenarioGARCH,
    generar_escenario_GARCH,
    resumen_escenario_GARCH,
)

# Escenario TAR (corrida 200): umbral suave sobre el rezago propio de cada score.
from .sim_escenario_TAR import (
    ConfigEscenarioTAR,
    generar_escenario_TAR,
    resumen_escenario_TAR,
    predictiva_oraculo_TAR,
)

# ══════════════════════════════════════════════════════════════════════════
# 2. CONTRATO DE ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════

from .artifacts import (
    ARCHIVOS,
    ArtefactosFPCA,
    ArtefactosODPC,
    nombre_dataset,
    guardar_curvas,
    cargar_curvas,
    cargar_curvas_true,
    guardar_representacion,
    cargar_representacion,
    guardar_fpca,
    cargar_fpca,
    guardar_estandarizador,
    cargar_estandarizador,
    guardar_datasets_ar,
    cargar_datasets_ar,
    guardar_hiperparametros,
    cargar_hiperparametros,
    guardar_config_evaluacion,
    cargar_config_evaluacion,
    guardar_draws_scores,
    cargar_draws_scores,
    verificar_contrato,
)

__all__ = [
    # ── Configuracion y salida de los generadores ──
    "ConfigObservacion",
    "SalidaSimulacion",
    # ── Grilla y funcion media ──
    "grilla_regular",
    "media_nula",
    "evaluar_media",
    # ── Cuadratura y operadores integrales ──
    "pesos_trapezoidales",
    "norma_hilbert_schmidt",
    "matriz_operador_ar",
    # ── Innovacion funcional gaussiana ──
    "matriz_covarianza_innovacion",
    "factor_cholesky",
    "generador_innovacion",
    # ── Reproducibilidad y esquema de observacion ──
    "semillas_replicas",
    "aplicar_ruido_observacion",
    # ── Control de calidad y persistencia de escenarios ──
    "diagnostico_comun",
    "guardar_escenario",
    "cargar_escenario",
    # ── Escenario 1: FAR(1) lineal gaussiano homogeneo ──
    "ConfigEscenario1",
    "generar_escenario_1",
    "resumen_escenario_1",
    "simular_trayectoria_far1",



    # ── Series de tiempo clasicas: mecanismo comun a A-1, A-2 y A-3 ──
    "acf_empirica_serie",
    "bloques_de_serie",
    "diagnostico_serie_escalar",
    "generar_serie_segmentada",
    "oraculo_lineal_un_rezago",
    "r2_empirico_media_condicional",





    # ── Algoritmos B-1, B-2 y B-3: mezcla de mecanismos funcionales (anexo, seccion B) ──
    "ConfigEscenarioB1",
    "generar_escenario_B1",
    "resumen_escenario_B1",
    "media_seno",
    "medias_mezcla",
    "funciones_rasgo",
    "calcular_rasgos",
    "transformar_identidad",
    "simular_mezcla_curvas",
    "ConfigEscenarioB2",
    "generar_escenario_B2",
    "resumen_escenario_B2",
    "transformar_rasgos_B2",
    "B_MECANISMOS_B2",
    "ConfigEscenarioB3",
    "generar_escenario_B3",
    "resumen_escenario_B3",
    "probabilidades_softmax",
    "A_ASIGNACION_B3",
    # ── Base de Fourier de los escenarios sobre scores ──
    "base_fourier",
    "gram_base",
    # ── Algoritmos C-1, C-2 y C-3: mezcla de mecanismos en el espacio de scores ──
    "ConfigEscenarioC1",
    "generar_escenario_C1",
    "resumen_escenario_C1",



    "ConfigEscenarioGARCH",
    "generar_escenario_GARCH",
    "resumen_escenario_GARCH",
    "ConfigEscenarioTAR",
    "generar_escenario_TAR",
    "resumen_escenario_TAR",
    "predictiva_oraculo_TAR",
    "MecanismoScores",
    "simular_mezcla_scores",
    "generar_mezcla_scores",
    "resumen_mezcla_scores",
    # ── Contrato de artefactos ──
    "ARCHIVOS",
    "ArtefactosFPCA",
    "ArtefactosODPC",
    "nombre_dataset",
    "guardar_curvas",
    "cargar_curvas",
    "cargar_curvas_true",
    "guardar_representacion",
    "cargar_representacion",
    "guardar_fpca",
    "cargar_fpca",
    "guardar_estandarizador",
    "cargar_estandarizador",
    "guardar_datasets_ar",
    "cargar_datasets_ar",
    "guardar_hiperparametros",
    "cargar_hiperparametros",
    "guardar_config_evaluacion",
    "cargar_config_evaluacion",
    "guardar_draws_scores",
    "cargar_draws_scores",
    "verificar_contrato",
]