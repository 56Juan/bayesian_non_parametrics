# Aplicacion de intervalos de credibilidad para estimacion de densidad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, laplace as laplace_dist, norm
from typing import Optional, Tuple, List

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_density_credible_interval(trace: dict,
                                   y_data: np.ndarray,
                                   kernel: str = 'normal',
                                   burn_in: int = 0,
                                   n_posterior: int = 2000,
                                   figsize: Tuple[int, int] = (12, 6),
                                   title: str = "Estimación de Densidad con Intervalo Credible",
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualiza la densidad estimada con intervalo credible del 95%.
    Compatible con kernels Normal y Laplace.
    
    Parameters
    ----------
    trace : dict
        Diccionario con las trazas
        - Normal: debe contener 'mu', 'sigma2', 'z', 'w'
        - Laplace: debe contener 'mu', 'b', 'z', 'w'
    y_data : np.ndarray
        Datos observados originales
    kernel : str
        Tipo de kernel: 'normal' o 'laplace'
    burn_in : int
        Iteraciones de burn-in a descartar (si no se aplicó antes)
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
    post_z = trace['z'][burn_in:]
    post_w = trace['w'][burn_in:]
    
    # Determinar parámetro de dispersión según kernel
    if kernel.lower() == 'normal':
        if 'sigma2' not in trace:
            raise KeyError("Trace debe contener 'sigma2' para kernel Normal")
        post_dispersion = trace['sigma2'][burn_in:]
    elif kernel.lower() == 'laplace':
        if 'b' not in trace:
            raise KeyError("Trace debe contener 'b' para kernel Laplace")
        post_dispersion = trace['b'][burn_in:]
    else:
        raise ValueError(f"Kernel '{kernel}' no reconocido. Use 'normal' o 'laplace'")
    
    n_post = len(post_mu)
    
    # Generar muestras de la mezcla por iteración
    simulated = []
    
    for t in range(n_post):
        samples_t = []
        mu_t = np.array(post_mu[t])
        disp_t = np.array(post_dispersion[t])
        w_t = np.array(post_w[t])
        
        # Muestrear de la mezcla usando los pesos
        for _ in range(n_posterior):
            # Elegir un cluster según los pesos w_t
            cluster_id = np.random.choice(len(w_t), p=w_t)
            mu_k = mu_t[cluster_id]
            disp_k = disp_t[cluster_id]
            
            # Generar muestra según el kernel
            if kernel.lower() == 'normal':
                sigma_k = np.sqrt(disp_k)
                sample = np.random.normal(mu_k, sigma_k)
            else:  # laplace
                sample = np.random.laplace(mu_k, disp_k)
            
            samples_t.append(sample)
        
        simulated.append(samples_t)
    
    simulated = np.array(simulated)  # shape: (n_post, n_posterior)
    
    # Estimación de densidad KDE
    x_min = min(y_data.min(), simulated.min()) - 3
    x_max = max(y_data.max(), simulated.max()) + 3
    x_grid = np.linspace(x_min, x_max, 500)
    
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
    
    ax.plot(x_grid, mean_kde, color='blue', linewidth=2.5, 
            label='Densidad media posterior')
    ax.fill_between(x_grid, lower_kde, upper_kde, color='blue', alpha=0.3, 
                     label='95% Intervalo Credible')
    ax.hist(y_data, bins=30, density=True, color='gray', alpha=0.3, 
            edgecolor='black', label='Histograma de datos')
    
    # Añadir info del kernel
    kernel_label = 'Normal' if kernel.lower() == 'normal' else 'Laplace'
    ax.text(0.02, 0.98, f'Kernel: {kernel_label}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Valor', fontsize=12)
    ax.set_ylabel('Densidad', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig