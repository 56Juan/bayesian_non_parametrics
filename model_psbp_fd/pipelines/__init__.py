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

# ══════════════════════════════════════════════════════════════════════════
# 2. CONTRATO DE ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════

from .artifacts import (
    ARCHIVOS,
    ArtefactosFPCA,
    nombre_dataset,
    guardar_curvas,
    cargar_curvas,
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
    # ── Escenario 4: innovaciones skew-normal (SMSN) ──
    # ── Escenario 5: proceso combinado ──
    # ── Contrato de artefactos ──
    "ARCHIVOS",
    "ArtefactosFPCA",
    "nombre_dataset",
    "guardar_curvas",
    "cargar_curvas",
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