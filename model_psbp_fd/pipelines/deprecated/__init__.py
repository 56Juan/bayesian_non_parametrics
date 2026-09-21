"""Generadores de corridas retiradas, conservados por si un escenario se reusa; NO forman parte del estudio oficial."""

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

# --- Escenario 3: FAR con cambio de regimen ---
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

# --- Escenario B: FAR con signo conmutado por umbral ---
from .sim_escenario_B import (
    ConfigEscenarioB,
    generar_escenario_B,
    resumen_escenario_B,
    simular_trayectoria_far_signo,
    direccion_oscilatoria,
    coeficiente_sarle_mezcla_simetrica,
)

# --- Escenarios C-F: tendencia + no linealidad intra-curva ---
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

# --- Escenario TS: familia T con cambio de simulador por umbral de nivel ---
from .sim_escenario_TS import (
    ConfigEscenarioTS,
    generar_escenario_TS,
    resumen_escenario_TS,
    trayectoria_nivel_acotada,
)

# --- Escenario J: cuatro rezagos, grado polinomial decreciente ---
from .sim_escenario_J import (
    ConfigEscenarioJ,
    generar_escenario_J,
    resumen_escenario_J,
)

# --- Escenario K: heterocedasticidad condicional en tau ---
from .sim_escenario_K import (
    ConfigEscenarioK,
    generar_escenario_K,
    resumen_escenario_K,
    envolvente_extremos_centro,
    simular_trayectoria_far_heterocedastico,
)

# --- Escenario L: mezcla de regresiones sobre coeficientes ---
from .sim_escenario_L import (
    ConfigEscenarioL,
    base_ortonormal,
    construir_regimenes,
    generar_escenario_L,
    resumen_escenario_L,
    simular_coeficientes,
)

# --- Escenario CE: composicion de estados en cuatro fases ---
from .sim_escenario_CE import (
    ConfigEscenarioCE,
    FASES,
    generar_escenario_CE,
    resumen_escenario_CE,
    simular_coeficientes_CE,
)

__all__ = [
    "ConfigEscenario2", "generar_escenario_2", "resumen_escenario_2",
    "simular_trayectoria_fgarch", "construir_operadores_garch",
    "matriz_kernel_no_negativo", "radio_espectral",
    "ConfigEscenario3", "generar_escenario_3", "resumen_escenario_3",
    "simular_trayectoria_far_regimen", "direccion_constante",
    "ConfigEscenario4", "generar_escenario_4", "resumen_escenario_4",
    "simular_trayectoria_far_smsn", "momentos_mezcla_escala",
    "extraer_factor_escala", "construir_componentes_smsn",
    "generador_innovacion_smsn",
    "ConfigEscenario5", "generar_escenario_5", "resumen_escenario_5",
    "simular_coeficientes_ar1", "base_fourier", "frecuencia_maxima_base",
    "espectro_geometrico", "phis_componente_predecible", "media_senoidal",
    "ConfigEscenario6", "generar_escenario_6", "resumen_escenario_6",
    "simular_coeficientes_ar1_no_estacionario", "trayectoria_espectro",
    "espectro_intercambiado",
    "ConfigEscenarioB", "generar_escenario_B", "resumen_escenario_B",
    "simular_trayectoria_far_signo", "direccion_oscilatoria",
    "coeficiente_sarle_mezcla_simetrica",
    "ConfigEscenarioT", "generar_escenario_T", "resumen_escenario_T",
    "perfil_tendencia", "perfil_tramos", "forma_tendencia_lineal_en_tau",
    "nucleo_local", "coeficiente_sarle_mezcla",
    "ConfigEscenarioTS", "generar_escenario_TS", "resumen_escenario_TS",
    "trayectoria_nivel_acotada",
    "ConfigEscenarioJ", "generar_escenario_J", "resumen_escenario_J",
    "ConfigEscenarioK", "generar_escenario_K", "resumen_escenario_K",
    "envolvente_extremos_centro", "simular_trayectoria_far_heterocedastico",
    "ConfigEscenarioL", "base_ortonormal", "construir_regimenes",
    "generar_escenario_L", "resumen_escenario_L", "simular_coeficientes",
    "ConfigEscenarioCE", "FASES", "generar_escenario_CE",
    "resumen_escenario_CE", "simular_coeficientes_CE",
]
