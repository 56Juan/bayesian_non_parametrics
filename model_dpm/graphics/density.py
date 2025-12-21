# Modulos de densidad

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from typing import Optional, Tuple, List

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_density_estimation(data: np.ndarray, 
                           bins: int = 50,
                           figsize: Tuple[int, int] = (14, 5),
                           title: str = "Estimación de Densidad",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza histograma y KDE de los datos.
    
    Parameters
    ----------
    data : np.ndarray
        Datos a visualizar
    bins : int
        Número de bins para el histograma
    figsize : tuple
        Tamaño de la figura
    title : str
        Título principal
    save_path : str, optional
        Ruta para guardar la figura
    
    Returns
    -------
    fig : plt.Figure
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histograma
    axes[0].hist(data, bins=bins, density=True, alpha=0.7, 
                 edgecolor='black', color='steelblue')
    axes[0].set_xlabel('Valor', fontsize=12)
    axes[0].set_ylabel('Densidad', fontsize=12)
    axes[0].set_title('Histograma', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # KDE
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    kde_vals = kde(x_range)
    
    axes[1].plot(x_range, kde_vals, linewidth=2.5, color='steelblue', label='KDE')
    axes[1].fill_between(x_range, kde_vals, alpha=0.3, color='steelblue')
    axes[1].set_xlabel('Valor', fontsize=12)
    axes[1].set_ylabel('Densidad', fontsize=12)
    axes[1].set_title('Densidad Estimada (KDE)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
