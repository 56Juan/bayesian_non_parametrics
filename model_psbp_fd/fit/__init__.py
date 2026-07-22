"""
fit
===
Evaluacion del desempeno predictivo y diagnostico de las cadenas MCMC.

Nota sobre el nombre del paquete: pese a llamarse `fit`, su contenido no ajusta
modelos ---eso ocurre en `models/` y en el muestreador de MATLAB--- sino que
evalua predicciones ya generadas. Los modulos comparan predicciones contra
observaciones sin asumir forma alguna para el mecanismo que las produce, de
modo que permiten contrastar en igualdad de condiciones modelos de naturaleza
distinta.

Organizacion, alineada con la Seccion 2.2.3 del marco teorico:

    metrics_puntual.py        Seccion 2.2.3.2 — error puntual en dos niveles:
                              coeficientes de la representacion y curva
                              reconstruida (MISE).
    metrics_distribucional.py Seccion 2.2.3.3 — reglas de puntuacion propias
                              (CRPS, puntaje de energia, puntaje logaritmico),
                              cobertura de intervalos y transformada integral
                              de probabilidad.
    pooling.py                Agrupacion de cadenas MCMC independientes en una
                              unica distribucion predictiva.
    baselines.py              Lineas base sobre el bloque de prueba: media
                              incondicional y persistencia.

Advertencia sobre las formas cerradas gaussianas
------------------------------------------------
`crps_gaussiano`, `lps_gaussiano` y `pit_gaussiano` implementan la
aproximacion de dos momentos de la distribucion predictiva. Es adecuada para
modelos cuya predictiva es efectivamente gaussiana, pero aplicarla a la
predictiva de mezcla del modelo propuesto descarta su forma ---multimodalidad,
asimetria--- que es precisamente aquello que el estudio busca medir. Para el
modelo de mezcla deben emplearse las versiones muestrales o
`lps_desde_log_densidad` con la densidad de la mezcla en forma cerrada.
"""

from .metrics_puntual import (
    rmse,
    mse_por_coeficiente,
    rmse_por_coeficiente,
    r2_por_columna,
    razon_dispersion,
    mise,
    rmse_funcional,
    resumen_puntual,
)

from .metrics_distribucional import (
    crps_muestral,
    crps_gaussiano,
    energy_score,
    cobertura,
    intervalo_muestral,
    pit_muestral,
    pit_gaussiano,
    diagnostico_pit,
    lps_gaussiano,
    lps_desde_log_densidad,
)

from .pooling import (
    agrupar_momentos,
    agrupar_muestras,
)

from .baselines import (
    prediccion_media_incondicional,
    prediccion_persistencia,
    evaluar_baseline,
    tabla_baselines,
)

__all__ = [
    # ── Error puntual (§2.2.3.2) ──
    "rmse",
    "mse_por_coeficiente",
    "rmse_por_coeficiente",
    "r2_por_columna",
    "razon_dispersion",
    "mise",
    "rmse_funcional",
    "resumen_puntual",
    # ── Distribucion predictiva (§2.2.3.3) ──
    "crps_muestral",
    "crps_gaussiano",
    "energy_score",
    "cobertura",
    "intervalo_muestral",
    "pit_muestral",
    "pit_gaussiano",
    "diagnostico_pit",
    "lps_gaussiano",
    "lps_desde_log_densidad",
    # ── Agrupacion de cadenas MCMC ──
    "agrupar_momentos",
    "agrupar_muestras",
    # ── Lineas base ──
    "prediccion_media_incondicional",
    "prediccion_persistencia",
    "evaluar_baseline",
    "tabla_baselines",
    # ── Diagnosticos MCMC (pendiente) ──
    # ESS, Geweke y R-hat viven hoy dentro de las funciones plot_convergence_*
    # de `graphics/`. Separar el calculo del dibujo permitiria reportar la
    # tabla de diagnosticos sin generar figuras; queda como `diagnostics_mcmc.py`.
]