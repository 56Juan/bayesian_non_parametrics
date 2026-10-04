"""
pipelines
=========
Composicion del estudio: generacion de datos, persistencia de artefactos y
contrato entre los flujos de trabajo.

El paquete agrupa tres responsabilidades:

1. Generadores de datos (`sim_comun.py`, `sim_escenario_k.py`).
   `sim_comun` aloja el esquema de observacion, la cuadratura de los operadores
   integrales, la innovacion funcional gaussiana y el control de calidad
   transversal. Cada `sim_escenario_k` implementa el Algoritmo k del anexo y
   expone el mismo contrato: una dataclass de configuracion que hereda de
   `ConfigObservacion`, una funcion generadora que retorna R replicas
   independientes, y una funcion de control de calidad especifica del
   algoritmo. El producto entregado a los metodos es siempre la matriz de
   observaciones discretas contaminadas con ruido de medicion, de dimension
   (R, T, L).

2. Contrato de persistencia (`artifacts.py`).
   Funciones emparejadas de escritura y lectura para los artefactos que el
   flujo de preprocesamiento produce y que el flujo de resultados y MATLAB
   consumen. Concentrar ambos extremos en un mismo modulo evita que la
   convencion de nombres y las garantias de forma se repliquen ---y diverjan---
   en cada notebook.

3. Verificacion cruzada (`verificar_contrato`).
   Comprueba la consistencia entre manifest, hiperparametros y artefactos FPCA
   antes de que el analisis comience.

Generadores vivos: `sim_comun`, `sim_scores_comun` (motor de los escenarios sobre
scores), `sim_escenario_TAR` (corrida 200), `sim_escenario_GARCH` (201) y
`sim_escenario_C1` (202, mezcla de mecanismos en el espacio de scores), mas
`sim_escenario_1/B1/B2/B3/C` y `sim_series_clasicas`, de los que dependen. Los
generadores de corridas retiradas (Escenarios 2-6, A-1, A-2, A-3, B, C-2, C-3, C-F,
TS, J, K, L y CE) viven en `pipelines/deprecated/`: siguen importables desde ahi,
no desde este paquete, y no forman parte del estudio oficial.
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

# --- Algoritmos C-1, C-2 y C-3: coeficientes de la representacion (seccion C) ---
from .sim_escenario_C import (
    ConfigEscenarioC,
    base_fourier,
    config_C1,
    config_C2,
    config_C3,
    contabilidad_varianzas,
    correlaciones_impulsor,
    curvas_truncadas,
    generar_escenario_C,
    resumen_escenario_C,
)
# --- Algoritmos C-1, C-2 y C-3 del anexo vigente: mezcla de mecanismos en el
#     espacio de scores (corridas 107-109). `sim_escenario_C` es otro diseno.
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

# Escenario TAR (corrida 114): umbral suave sobre el rezago propio de cada score.
from .sim_escenario_TAR import (
    ConfigEscenarioTAR,
    generar_escenario_TAR,
    resumen_escenario_TAR,
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
    # ── Algoritmos C-1, C-2 y C-3: coeficientes de la representacion ──
    "ConfigEscenarioC",
    "config_C1",
    "config_C2",
    "config_C3",
    "generar_escenario_C",
    "resumen_escenario_C",
    "base_fourier",
    "curvas_truncadas",
    "contabilidad_varianzas",
    "correlaciones_impulsor",
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
    "verificar_contrato",
]