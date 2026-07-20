"""
Generadores de datos del estudio de simulacion.

El modulo `sim_comun.py` aloja el esquema de observacion, la cuadratura de los
operadores integrales, la innovacion funcional gaussiana y el control de
calidad transversal a todos los escenarios.

Cada modulo `sim_escenario_k.py` implementa el Algoritmo k del anexo y expone
el mismo contrato: una dataclass de configuracion que hereda de
`ConfigObservacion`, una funcion generadora que retorna R replicas
independientes, y una funcion de control de calidad especifica del algoritmo.
El producto entregado a los metodos es siempre la matriz de observaciones
discretas contaminadas con ruido de medicion, de dimension (R, T, L).
"""

# --- Componentes transversales ---
from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    grilla_regular,
    media_nula,
    evaluar_media,
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

__all__ = [
    # --- Configuracion y salida (transversal) ---
    "ConfigObservacion",
    "SalidaSimulacion",
    # --- Grilla y funcion media ---
    "grilla_regular",
    "media_nula",
    "evaluar_media",
    # --- Operadores integrales y cuadratura ---
    "norma_hilbert_schmidt",
    "matriz_operador_ar",
    # --- Innovacion funcional gaussiana ---
    "matriz_covarianza_innovacion",
    "factor_cholesky",
    "generador_innovacion",
    # --- Reproducibilidad y esquema de observacion ---
    "semillas_replicas",
    "aplicar_ruido_observacion",
    # --- Control de calidad y persistencia ---
    "diagnostico_comun",
    "guardar_escenario",
    "cargar_escenario",
    # --- Escenario 1: FAR(1) lineal gaussiano homogeneo ---
    "ConfigEscenario1",
    "generar_escenario_1",
    "resumen_escenario_1",
    "simular_trayectoria_far1",
    # --- Escenario 2: FGARCH(1,1) ---
    # --- Escenario 3: FAR con cambio de regimen ---
    # --- Escenario 4: innovaciones skew-normal (SMSN) ---
    # --- Escenario 5: proceso combinado ---
]