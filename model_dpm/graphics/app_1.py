# Aplicacion de intervalos de credibilidad para estimacion de densidad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from typing import Optional, Tuple, List

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_density_credible_interval(trace: dict,
                                   y_data: np.ndarray,
                                   burn_in: int = 300,
                                   n_posterior: int = 2000,
                                   figsize: Tuple[int, int] = (12, 6),
                                   title: str = "Estimación de Densidad con Intervalo Credible",
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza la densidad estimada con intervalo credible del 95%.
    
    Parameters
    ----------
    trace : dict
        Diccionario con las trazas (debe contener 'mu', 'sigma2', 'z')
    y_data : np.ndarray
        Datos observados originales
    burn_in : int
        Iteraciones de burn-in a descartar
    n_posterior : int
        Número de muestras a generar por iteración
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
    # Extraer trazas post burn-in
    post_mu = trace['mu'][burn_in:]
    post_sigma2 = trace['sigma2'][burn_in:]
    post_z = trace['z'][burn_in:]
    n_post = len(post_mu)
    
    # Generar muestras de la mezcla por iteración
    simulated = []
    
    for t in range(n_post):
        samples_t = []
        mu_t = np.array(post_mu[t])
        sigma2_t = np.array(post_sigma2[t])
        z_t = np.array(post_z[t])
        
        for _ in range(n_posterior):
            # Elegir un cluster al azar según las asignaciones z
            cluster_id = np.random.choice(z_t)
            mu_k = mu_t[cluster_id]
            sigma_k = np.sqrt(sigma2_t[cluster_id])
            samples_t.append(np.random.normal(mu_k, sigma_k))
        
        simulated.append(samples_t)
    
    simulated = np.array(simulated)  # shape: (n_post, n_posterior)
    
    # Estimación de densidad KDE
    x_grid = np.linspace(y_data.min() - 3, y_data.max() + 3, 500)
    kde_matrix = []
    
    for t in range(n_post):
        data = simulated[t, :]
        kde = gaussian_kde(data, bw_method='scott')
        kde_matrix.append(kde(x_grid))
    
    kde_matrix = np.array(kde_matrix)
    
    # Calcular estadísticas
    mean_kde = np.mean(kde_matrix, axis=0)
    lower_kde = np.percentile(kde_matrix, 2.5, axis=0)
    upper_kde = np.percentile(kde_matrix, 97.5, axis=0)
    
    # Graficar
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x_grid, mean_kde, color='blue', linewidth=2.5, label='Densidad media posterior')
    ax.fill_between(x_grid, lower_kde, upper_kde, color='blue', alpha=0.3, 
                     label='95% Intervalo Credible')
    ax.hist(y_data, bins=30, density=True, color='gray', alpha=0.3, 
            edgecolor='black', label='Histograma de datos')
    
    ax.set_xlabel('Valor', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig