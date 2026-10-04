from .viz_traces import (
    plot_traces_bj,
    plot_traces_pj,
    plot_convergence_bj,
    plot_convergence_pj,
)

from .viz_global_components import (
    plot_global_components,
    plot_active_clusters,
)

from .viz_functional_data import (
    plot_functional_series,
    plot_empirical_sample,
    plot_functional_mean,
    plot_functional_variance,
    plot_mean_and_variance,
)

from .viz_time_series import (
    plot_fts_empirical,
    plot_fts_functional,
    plot_fts_comparison,
)

from .viz_prediction import (
    plot_scatter_theta,
    plot_functional_comparison,
)

from .viz_preprocessing import (
    plot_diagnostico_estandarizacion,
    plot_seleccion_basis,
)

from .viz_fpca import (
    plot_fpca_scree,
    plot_fpca_correlacion_lag0,
    plot_rezagos_heatmap,
    plot_series_componentes,
    plot_diagnostico_rezagos,
)

from .viz_convergencia import (
    plot_ocupacion_mezcla,
    plot_gating,
    plot_diagnostico_vs_M,
    plot_ocupacion_vs_M,
    plot_pip_vs_M,
    plot_variantes,
)

from .viz_comparacion import (
    ESTILOS_MODELOS,
    plot_ganador_modelo,
    plot_metricas_vs_M,
    plot_bandas_contraste,
)

from .viz_evaluacion_barrido import (
    plot_ganador_ventana,
    plot_scores_banda,
    plot_scores_dispersion,
)

from .viz_evaluacion import (
    plot_ventana_movil,
    plot_bandas_serie,
    plot_extractos_curvas,
    plot_calibracion_pit,
)

__all__ = [
    # viz_traces
    "plot_traces_bj",
    "plot_traces_pj",
    "plot_convergence_bj",
    "plot_convergence_pj",
    # viz_global_components
    "plot_global_components",
    "plot_active_clusters",
    # viz_functional_data
    "plot_functional_series",
    "plot_empirical_sample",
    "plot_functional_mean",
    "plot_functional_variance",
    "plot_mean_and_variance",
    # viz_time_series
    "plot_fts_empirical",
    "plot_fts_functional",
    "plot_fts_comparison",
    # viz_prediction
    "plot_scatter_theta",
    "plot_functional_comparison",
    # viz_preprocessing
    "plot_diagnostico_estandarizacion",
    "plot_seleccion_basis",
    # viz_fpca
    "plot_fpca_scree",
    "plot_fpca_correlacion_lag0",
    "plot_rezagos_heatmap",
    "plot_series_componentes",
    "plot_diagnostico_rezagos",
    # viz_comparacion
    "ESTILOS_MODELOS",
    "plot_ganador_modelo",
    "plot_metricas_vs_M",
    "plot_bandas_contraste",
    # viz_evaluacion_barrido
    "plot_ganador_ventana",
    "plot_scores_banda",
    "plot_scores_dispersion",
    # viz_evaluacion
    # viz_convergencia
    "plot_ocupacion_mezcla",
    "plot_gating",
    "plot_diagnostico_vs_M",
    "plot_ocupacion_vs_M",
    "plot_pip_vs_M",
    "plot_variantes",
    # viz_evaluacion
    "plot_ventana_movil",
    "plot_bandas_serie",
    "plot_extractos_curvas",
    "plot_calibracion_pit",
]
