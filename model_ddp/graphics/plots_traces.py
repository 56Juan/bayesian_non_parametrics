import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_hyperparameter_traces(trace, param_config, output_path, title="Trazas de Hiperparámetros"):
    """
    Genera gráficas de trazas para hiperparámetros de modelos DDP.
    
    Parámetros:
    -----------
    trace : dict o objeto con atributos
        Objeto con las trazas MCMC. Debe permitir acceso como trace[param_name]
        
    param_config : list of tuples
        Lista de tuplas (nombre_parametro, etiqueta_display)
        Ejemplo: [('mu', 'μ (Intercepto)'), ('sigma', 'σ²')]
        
    output_path : str or Path
        Ruta donde guardar la imagen (ej: "carpeta/trazas.png")
        
    title : str, opcional
        Título general de la figura
        
    Retorna:
    --------
    fig : matplotlib.figure.Figure
        Objeto figura (por si se quiere manipular después)
    """
    n_params = len(param_config)
    
    # Calcular grid óptimo (prioriza filas de 3 columnas)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Asegurar que axes sea siempre un array 2D
    if n_params == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plotear cada parámetro
    for idx, (param_name, param_label) in enumerate(param_config):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Extraer datos de la traza
        try:
            param_trace = trace[param_name]
        except (KeyError, TypeError, AttributeError):
            ax.text(0.5, 0.5, f'Parámetro "{param_name}"\nno encontrado', 
                   ha='center', va='center', fontsize=10, color='red')
            ax.set_title(param_label, fontsize=10)
            continue
        
        # Plotear
        ax.plot(param_trace, linewidth=1, alpha=0.8)
        ax.set_xlabel('Iteración', fontsize=10)
        ax.set_ylabel(param_label, fontsize=10)
        ax.grid(alpha=0.3)
        
        # Calcular estadísticas
        mean_val = np.mean(param_trace)
        ax.set_title(f'{param_label}\nMedia: {mean_val:.3f}', fontsize=10)
    
    # Ocultar ejes sobrantes si n_params no llena el grid
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


# ============================================================
# EJEMPLOS DE USO PARA DIFERENTES MODELOS DDP
# ============================================================

# Ejemplo 1: Modelo básico DDP
#hyperparams_basic = [
#    ('mu', 'μ (Intercepto stick-breaking)'),
#    ('mu0', 'μ₀ (Media base)'),
#    ('kappa0', 'κ₀ (Precisión relativa)'),
#    ('a0', 'a₀ (Shape σ²)'),
#    ('b0', 'b₀ (Scale σ²)'),
#    ('n_clusters', 'Número de Clusters')
#]

# plot_hyperparameter_traces(
#     trace=trace,
#     param_config=hyperparams_basic,
#     output_path=carpeta_graficas / "trazas_hiperparametros.png"
# )


## Ejemplo 2: Modelo DDP con covariables
#hyperparams_covariates = [
#    ('mu', 'μ (Intercepto)'),
#    ('beta_age', 'β (Edad)'),
#    ('beta_gender', 'β (Género)'),
#    ('sigma_beta', 'σ_β (Varianza coeficientes)'),
#    ('alpha', 'α (Concentración DP)'),
#    ('n_clusters', 'Número de Clusters')
#]

# plot_hyperparameter_traces(
#     trace=trace,
#     param_config=hyperparams_covariates,
#     output_path=carpeta_graficas / "trazas_covariates.png",
#     title="Trazas DDP con Covariables"
# )
