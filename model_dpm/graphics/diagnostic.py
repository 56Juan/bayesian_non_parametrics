# Modulos de analisis de la aplicacion
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from typing import Optional, Tuple, List

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_cluster_assignment(data: np.ndarray,
                           clusters: np.ndarray,
                           figsize: Tuple[int, int] = (12, 6),
                           cmap: str = 'viridis',
                           title: str = "Asignación de Clusters",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza la asignación de clusters a lo largo del índice de observación.
    
    Parameters
    ----------
    data : np.ndarray
        Valores de los datos
    clusters : np.ndarray
        Asignación de clusters para cada observación
    figsize : tuple
        Tamaño de la figura
    cmap : str
        Colormap para los clusters
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    
    Returns
    -------
    fig : plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(range(len(data)), data, 
                        c=clusters, cmap=cmap, 
                        alpha=0.6, edgecolor='black', linewidth=0.5, s=50)
    
    ax.set_xlabel('Índice de observación', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
    cbar.set_label('Cluster', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cluster_densities(data: np.ndarray,
                          clusters: np.ndarray,
                          figsize: Tuple[int, int] = (12, 6),
                          title: str = "Densidad por Cluster",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza la densidad KDE de cada cluster en un mismo gráfico.
    
    Parameters
    ----------
    data : np.ndarray
        Valores de los datos
    clusters : np.ndarray
        Asignación de clusters
    figsize : tuple
        Tamaño de la figura
    title : str
        Título del gráfico
    save_path : str, optional
        Ruta para guardar la figura
    
    Returns
    -------
    fig : plt.Figure
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_clusters = np.unique(clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
    
    # Graficar densidad para cada cluster
    for i, cluster in enumerate(sorted(unique_clusters)):
        cluster_data = data[clusters == cluster]
        
        if len(cluster_data) > 1:  # KDE necesita al menos 2 puntos
            kde = gaussian_kde(cluster_data)
            x_range = np.linspace(cluster_data.min(), cluster_data.max(), 200)
            kde_vals = kde(x_range)
            
            ax.plot(x_range, kde_vals, linewidth=2.5, 
                   color=colors[i], label=f'Cluster {cluster} (n={len(cluster_data)})')
            ax.fill_between(x_range, kde_vals, alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Valor', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_traces(trace: dict,
               burn_in: int = 0,
               figsize: Tuple[int, int] = (12, 18),
               title: str = "Trazas del Modelo DPM",
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza las trazas de los hiperparámetros y clusters activos.
    
    Parameters
    ----------
    trace : dict
        Diccionario con las trazas del modelo (debe contener: 'M', 'mu0', 
        'kappa0', 'a0', 'b0', 'n_clusters', 'z', 'mu')
    burn_in : int
        Número de iteraciones de burn-in a descartar en la visualización
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
    fig, axes = plt.subplots(6, 1, figsize=figsize)
    
    # M (concentración)
    axes[0].plot(trace['M'][burn_in:], linewidth=1, color='steelblue')
    axes[0].axhline(np.mean(trace['M'][burn_in:]), color='red', 
                    linestyle='--', linewidth=2, label=f"Media: {np.mean(trace['M'][burn_in:]):.2f}")
    axes[0].set_ylabel('M', fontsize=12)
    axes[0].set_title('Parámetro de Concentración (M)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # μ₀
    axes[1].plot(trace['mu0'][burn_in:], linewidth=1, color='steelblue')
    axes[1].axhline(np.mean(trace['mu0'][burn_in:]), color='red', 
                    linestyle='--', linewidth=2, label=f"Media: {np.mean(trace['mu0'][burn_in:]):.2f}")
    axes[1].set_ylabel('μ₀', fontsize=12)
    axes[1].set_title('Hiperparámetro μ₀', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # κ₀
    axes[2].plot(trace['kappa0'][burn_in:], linewidth=1, color='steelblue')
    axes[2].axhline(np.mean(trace['kappa0'][burn_in:]), color='red', 
                    linestyle='--', linewidth=2, label=f"Media: {np.mean(trace['kappa0'][burn_in:]):.2f}")
    axes[2].set_ylabel('κ₀', fontsize=12)
    axes[2].set_title('Hiperparámetro κ₀', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # a₀
    axes[3].plot(trace['a0'][burn_in:], linewidth=1, color='steelblue')
    axes[3].axhline(np.mean(trace['a0'][burn_in:]), color='red', 
                    linestyle='--', linewidth=2, label=f"Media: {np.mean(trace['a0'][burn_in:]):.2f}")
    axes[3].set_ylabel('a₀', fontsize=12)
    axes[3].set_title('Hiperparámetro a₀', fontsize=12, fontweight='bold')
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    # Clusters activos
    axes[4].plot(trace['n_clusters'][burn_in:], linewidth=1.5, color='darkgreen')
    axes[4].axhline(np.mean(trace['n_clusters'][burn_in:]), color='red', 
                    linestyle='--', linewidth=2, label=f"Media: {np.mean(trace['n_clusters'][burn_in:]):.1f}")
    axes[4].set_ylabel('K efectivo', fontsize=12)
    axes[4].set_title('Número de Clusters Activos', fontsize=12, fontweight='bold')
    axes[4].legend(fontsize=10)
    axes[4].grid(True, alpha=0.3)
    
    # Promedio de μ de clusters activos
    mu_active_mean = []
    for z, mu in zip(trace['z'][burn_in:], trace['mu'][burn_in:]):
        counts = np.array([np.sum(z == k) for k in range(len(mu))])
        active_mask = counts > 0
        if active_mask.sum() > 0:
            mu_active_mean.append(mu[active_mask].mean())
        else:
            mu_active_mean.append(np.nan)
    
    axes[5].plot(mu_active_mean, linewidth=1, color='steelblue')
    axes[5].axhline(np.nanmean(mu_active_mean), color='red', 
                    linestyle='--', linewidth=2, label=f"Media: {np.nanmean(mu_active_mean):.2f}")
    axes[5].set_ylabel('μ promedio', fontsize=12)
    axes[5].set_title('Promedio de μ en Clusters Activos', fontsize=12, fontweight='bold')
    axes[5].legend(fontsize=10)
    axes[5].grid(True, alpha=0.3)
    
    for ax in axes:
        ax.set_xlabel('Iteración', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig