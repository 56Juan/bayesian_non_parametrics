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

Los seis algoritmos del anexo estan implementados. Los Algoritmos 1 a 4
(primer bloque) intervienen la ley condicional y operan sobre la grilla
mediante operadores integrales. Los Algoritmos 5 y 6 (segundo bloque) dejan la
ley condicional dentro de lo representable y comprometen la reduccion de
dimension; se definen sobre los coeficientes de un sistema ortonormal fijo, de
modo que comparten con los anteriores el esquema de observacion, las replicas y
las semillas, pero NO la maquinaria de operadores integrales, cuadratura de la
dinamica ni factorizacion de la innovacion funcional.

Ademas de los seis, `sim_escenario_B.py` implementa un escenario de DIAGNOSTICO
---no es un Algoritmo del anexo, igual que el Escenario A de la corrida 17---
construido para que la clase lineal homogenea falle: un FAR(1) cuyo operador
cambia de SIGNO segun un umbral sobre el estado rezagado. La antisimetria de
los dos regimenes cancela la covarianza cruzada, de modo que el mejor predictor
lineal no supera a la media incondicional mientras un tercio de la varianza
sigue siendo predecible. Es el unico escenario del estudio en que el MISE
discrimina entre la clase lineal y un metodo capaz de representar una media
condicional no lineal.
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

# --- Escenario 2: FGARCH(1,1) ---
from .sim_escenario_2 import (
    ConfigEscenario2,
    generar_escenario_2,
    resumen_escenario_2,
    simular_trayectoria_fgarch,
    construir_operadores_garch,
    matriz_kernel_no_negativo,
    radio_espectral,
)

# --- Escenario 3: FAR con cambio de regimen (multimodalidad condicional) ---
from .sim_escenario_3 import (
    ConfigEscenario3,
    generar_escenario_3,
    resumen_escenario_3,
    simular_trayectoria_far_regimen,
    direccion_constante,
)

# --- Escenario 4: innovaciones skew-normal (SMSN) ---
from .sim_escenario_4 import (
    ConfigEscenario4,
    generar_escenario_4,
    resumen_escenario_4,
    simular_trayectoria_far_smsn,
    momentos_mezcla_escala,
    extraer_factor_escala,
    construir_componentes_smsn,
    generador_innovacion_smsn,
)

# --- Escenario 5: predictibilidad en componente subordinada ---
from .sim_escenario_5 import (
    ConfigEscenario5,
    generar_escenario_5,
    resumen_escenario_5,
    simular_coeficientes_ar1,
    base_fourier,
    frecuencia_maxima_base,
    espectro_geometrico,
    phis_componente_predecible,
    media_senoidal,
)

# --- Escenario 6: covarianza no estacionaria ---
from .sim_escenario_6 import (
    ConfigEscenario6,
    generar_escenario_6,
    resumen_escenario_6,
    simular_coeficientes_ar1_no_estacionario,
    trayectoria_espectro,
    espectro_intercambiado,
)

# --- Escenario B: FAR con signo conmutado por umbral (diagnostico) ---
from .sim_escenario_B import (
    ConfigEscenarioB,
    generar_escenario_B,
    resumen_escenario_B,
    simular_trayectoria_far_signo,
    direccion_oscilatoria,
    coeficiente_sarle_mezcla_simetrica,
)

# --- Escenarios C-F: tendencia + no linealidad intra-curva (diagnostico) ---
from .sim_escenario_T import (
    ConfigEscenarioT,
    generar_escenario_T,
    resumen_escenario_T,
    perfil_tendencia,
    perfil_tramos,
    forma_tendencia_lineal_en_tau,
    nucleo_local,
    coeficiente_sarle_mezcla,
)

# ══════════════════════════════════════════════════════════════════════════
# 2. CONTRATO DE ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════

from .artifacts import (
    ARCHIVOS,
    ArtefactosFPCA,
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
    "matriz_kernel_no_negativo",
    "radio_espectral",
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
    # ── Escenario 2: FGARCH(1,1) ──
    "ConfigEscenario2",
    "generar_escenario_2",
    "resumen_escenario_2",
    "simular_trayectoria_fgarch",
    "construir_operadores_garch",
    # ── Escenario 3: FAR con cambio de regimen ──
    "ConfigEscenario3",
    "generar_escenario_3",
    "resumen_escenario_3",
    "simular_trayectoria_far_regimen",
    "direccion_constante",
    # ── Escenario 4: innovaciones skew-normal (SMSN) ──
    "ConfigEscenario4",
    "generar_escenario_4",
    "resumen_escenario_4",
    "simular_trayectoria_far_smsn",
    "momentos_mezcla_escala",
    "extraer_factor_escala",
    "construir_componentes_smsn",
    "generador_innovacion_smsn",
    # ── Escenario 5: predictibilidad en componente subordinada ──
    "ConfigEscenario5",
    "generar_escenario_5",
    "resumen_escenario_5",
    "simular_coeficientes_ar1",
    "base_fourier",
    "frecuencia_maxima_base",
    "espectro_geometrico",
    "phis_componente_predecible",
    "media_senoidal",
    # ── Escenario 6: covarianza no estacionaria ──
    "ConfigEscenario6",
    "generar_escenario_6",
    "resumen_escenario_6",
    "simular_coeficientes_ar1_no_estacionario",
    "trayectoria_espectro",
    "espectro_intercambiado",
    # ── Escenario B: FAR con signo conmutado por umbral (diagnostico) ──
    "ConfigEscenarioB",
    "generar_escenario_B",
    "resumen_escenario_B",
    "simular_trayectoria_far_signo",
    "direccion_oscilatoria",
    "coeficiente_sarle_mezcla_simetrica",
    # ── Escenarios C-F: tendencia + no linealidad intra-curva (diagnostico) ──
    "ConfigEscenarioT",
    "generar_escenario_T",
    "resumen_escenario_T",
    "perfil_tendencia",
    "perfil_tramos",
    "forma_tendencia_lineal_en_tau",
    "nucleo_local",
    "coeficiente_sarle_mezcla",
    # ── Contrato de artefactos ──
    "ARCHIVOS",
    "ArtefactosFPCA",
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