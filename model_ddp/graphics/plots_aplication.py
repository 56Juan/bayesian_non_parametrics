import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paleta de colores moderna
COLORS = {
    'observed': '#2E86AB',      # Azul
    'predicted': '#A23B72',     # Púrpura
    'interval_95': '#F18F01',   # Naranja
    'interval_50': '#C73E1D',   # Rojo oscuro
}


def plot_credible_intervals(
    y_true,
    y_pred_mean,
    y_pred_std,
    output_path,
    sort_by='y_true',
    interval=0.95,
    title="Predicciones con Intervalos de Credibilidad",
    xlabel="Observación (ordenada)",
    ylabel="Y",
    figsize=(14, 7)
):
    """
    Visualiza E[Y|X] con intervalo de credibilidad bayesiano (simple y limpio).
    
    Parámetros:
    -----------
    y_true : array-like
        Valores observados (ground truth)
    
    y_pred_mean : array-like
        Media predictiva posterior E[Y|X, datos]
    
    y_pred_std : array-like
        Desviación estándar predictiva posterior σ[Y|X, datos]
    
    output_path : str or Path
        Ruta donde guardar la gráfica
    
    sort_by : str, opcional
        Criterio de ordenamiento: 'y_true', 'y_pred', 'index', 'residual'
    
    interval : float, opcional
        Nivel de credibilidad (0.95 para 95%, 0.68 para 68%, etc.)
    
    Interpretación:
    ---------------
    - Puntos azules: Valores reales observados Y
    - Línea púrpura: Esperanza condicional E[Y|X, datos]
    - Banda naranja: Intervalo de credibilidad (donde caerá Y con prob. = interval)
    """
    
    output_path = Path(output_path)
    
    # Convertir a arrays
    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)
    y_pred_std = np.asarray(y_pred_std)
    
    # Ordenamiento
    if sort_by == 'y_true':
        sort_idx = np.argsort(y_true)
    elif sort_by == 'y_pred':
        sort_idx = np.argsort(y_pred_mean)
    elif sort_by == 'residual':
        residuals = np.abs(y_true - y_pred_mean)
        sort_idx = np.argsort(residuals)
    else:  # 'index'
        sort_idx = np.arange(len(y_true))
    
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred_mean[sort_idx]
    y_std_sorted = y_pred_std[sort_idx]
    
    x_axis = np.arange(len(y_true))
    
    # =========================
    # GRÁFICO ÚNICO Y LIMPIO
    # =========================
    fig, ax = plt.subplots(figsize=figsize)
    
    # Z-score según el intervalo
    z_scores = {0.95: 1.96, 0.90: 1.645, 0.68: 1.0, 0.50: 0.674}
    z = z_scores.get(interval, 1.96)
    
    # Intervalo de credibilidad (banda)
    ax.fill_between(
        x_axis,
        y_pred_sorted - z * y_std_sorted,
        y_pred_sorted + z * y_std_sorted,
        alpha=0.25,
        color=COLORS['interval_95'],
        label=f'IC {int(interval*100)}%',
        edgecolor='none'
    )
    
    # Línea de predicción E[Y|X]
    ax.plot(
        x_axis, 
        y_pred_sorted, 
        color=COLORS['predicted'],
        linewidth=2.5,
        label='E[Y|X, datos]',
        alpha=0.9,
        zorder=5
    )
    
    # Puntos observados
    ax.scatter(
        x_axis,
        y_true_sorted,
        alpha=0.6,
        s=35,
        color=COLORS['observed'],
        label='Y observado',
        edgecolors='white',
        linewidth=0.5,
        zorder=6
    )
    
    # Estilo limpio
    ax.set_xlabel(xlabel, fontsize=12, fontweight='semibold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    
    # Leyenda elegante
    legend = ax.legend(
        loc='best',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=11
    )
    legend.get_frame().set_alpha(0.95)
    
    # Guardar
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
    
    # Métricas de calibración
    residuals = y_true - y_pred_mean
    coverage = np.mean(np.abs(residuals) <= z * y_pred_std)
    
    return {
        'coverage': coverage,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }


