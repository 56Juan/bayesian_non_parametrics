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
    rolling.py                Evolucion de las metricas sobre una ventana
                              movil que recorre entrenamiento y prueba, sin
                              reentrenar el modelo.
    inclusion.py              Probabilidades posteriores de inclusion (PIP) y
                              su contraste con la estructura del generador.
    diagnostics_mcmc.py       ESS, Geweke y R-hat separados del dibujo.
    pooling.py                Agrupacion de cadenas MCMC independientes en una
                              unica distribucion predictiva.
    baselines.py              Lineas base sobre el bloque de prueba: media
                              incondicional y persistencia.
    intervalos.py             Intervalo de prediccion a un paso IMPLICADO POR
                              EL MODELO, para competidores que solo entregan
                              prediccion puntual. En el FAR de Bosq la ley
                              condicional esta especificada por el propio
                              metodo, de modo que su intervalo es el clasico
                              del AR y no una banda adosada desde fuera.
    incertidumbre.py          Bootstrap de BLOQUES para la incertidumbre de una
                              metrica agregada sobre origenes dependientes. Es
                              modulo aparte de `pooling.py` a proposito: alli
                              la unidad de remuestreo es la extraccion del
                              posterior y aqui el origen de prediccion.

Las dos jerarquias de metricas, y por que estan escritas aqui
-------------------------------------------------------------
Con siete metricas siempre existe una bajo la cual el modelo preferido gana, de
modo que cual es PRIMARIA se fija ANTES de ver resultados y las tablas lo
declaran:

    Bloque A, error puntual --tres normas del MISMO error e_t(tau)--
        PRIMARIA     ||e||_2 (RMSE / MISE) si la decision pide la media
                     condicional, que es el funcional que el PSBPM-FD estima;
                     ||e||_1 (MAE) si pide la mediana. NO las dos a la vez: con
                     predictivas asimetricas pueden ordenar distinto.
        DIAGNOSTICAS ||e||_inf, el cuantil Q_0.95(|e|) y la razon
                     ||e||_inf/||e||_1, que mide concentracion del error.

    Bloque B, intervalos
        PRIMARIA     Winkler (interval score): regla de puntuacion propia, no
                     manipulable ensanchando ni estrechando el intervalo.
        DIAGNOSTICAS indicador de cobertura desagregado, PICP y MPIW. Son la
                     descomposicion del Winkler: cuando este es malo, dicen si
                     fue por cobertura insuficiente o por ancho excesivo.

Ninguna metrica esta ESCALADA (no hay MASE ni RMSSE): las comparaciones valen
dentro de una misma serie, no entre series de magnitudes distintas. Y los
errores no son independientes entre origenes --menos aun con ventanas
solapadas--, de modo que toda cuantificacion de incertidumbre sale del
bootstrap de bloques de `incertidumbre.py` y no del error estandar
independiente.

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
    mae,
    mae_por_coeficiente,
    cuantil_error_absoluto,
    error_maximo,
    pesos_normalizados,
    normas_error_por_origen,
    resumen_error_funcional,
)

from .metrics_distribucional import (
    crps_muestral,
    crps_gaussiano,
    energy_score,
    cobertura,
    estratos_por_cuantil,
    cobertura_condicional,
    intervalo_muestral,
    pit_muestral,
    pit_gaussiano,
    diagnostico_pit,
    lps_gaussiano,
    lps_desde_log_densidad,
    winkler,
    indicador_cobertura,
    indicador_cobertura_simultanea,
    picp,
    mpiw,
    resumen_intervalo,
)

from .rolling import (
    indices_ventanas,
    ventana_movil,
    ventana_movil_scores,
    ventana_movil_funcional,
)

from .inclusion import (
    pip_global,
    pip_por_componente,
    matriz_pip,
    contraste_con_verdad,
)

from .diagnostics_mcmc import (
    autocorr,
    ess_geyer,
    geweke_z,
    gelman_rubin,
    extraer_traza_variable,
    diagnostico_variable,
    tabla_diagnosticos,
    resumen_convergencia,
)

from .pooling import (
    agrupar_momentos,
    agrupar_muestras,
)

from .intervalos import (
    cuantil_normal,
    sigma_residual,
    residuos_para_banda,
    banda_predictiva_modelo,
)

from .incertidumbre import (
    largo_bloque_sugerido,
    bloques_circulares,
    bootstrap_bloques,
    diagnostico_dependencia,
    tabla_bootstrap,
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
    # -- Bloque A: normas L^p del error --
    "mae",
    "mae_por_coeficiente",
    "cuantil_error_absoluto",
    "error_maximo",
    "pesos_normalizados",
    "normas_error_por_origen",
    "resumen_error_funcional",
    # ── Distribucion predictiva (§2.2.3.3) ──
    "crps_muestral",
    "crps_gaussiano",
    "energy_score",
    "cobertura",
    "estratos_por_cuantil",
    "cobertura_condicional",
    "intervalo_muestral",
    "pit_muestral",
    "pit_gaussiano",
    "diagnostico_pit",
    "lps_gaussiano",
    "lps_desde_log_densidad",
    # -- Bloque B: intervalos de prediccion --
    "winkler",
    "indicador_cobertura",
    "indicador_cobertura_simultanea",
    "picp",
    "mpiw",
    "resumen_intervalo",
    # ── Ventana movil (§03_06 eje 1) ──
    "indices_ventanas",
    "ventana_movil",
    "ventana_movil_scores",
    "ventana_movil_funcional",
    # ── Probabilidades de inclusion (§03_06 eje 3) ──
    "pip_global",
    "pip_por_componente",
    "matriz_pip",
    "contraste_con_verdad",
    # ── Diagnosticos MCMC ──
    "autocorr",
    "ess_geyer",
    "geweke_z",
    "gelman_rubin",
    "extraer_traza_variable",
    "diagnostico_variable",
    "tabla_diagnosticos",
    "resumen_convergencia",
    # ── Agrupacion de cadenas MCMC ──
    "agrupar_momentos",
    "agrupar_muestras",
    # ── Intervalos implicados por el modelo ──
    "cuantil_normal",
    "sigma_residual",
    "residuos_para_banda",
    "banda_predictiva_modelo",
    # ── Incertidumbre bajo dependencia ──
    "largo_bloque_sugerido",
    "bloques_circulares",
    "bootstrap_bloques",
    "diagnostico_dependencia",
    "tabla_bootstrap",
    # ── Lineas base ──
    "prediccion_media_incondicional",
    "prediccion_persistencia",
    "evaluar_baseline",
    "tabla_baselines",
]