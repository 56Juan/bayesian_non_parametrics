import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from pathlib import Path

# Configuración de estilo global
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'real': '#2E86AB',      # Azul profesional
    'pred': '#A23B72',      # Púrpura
    'perfect': '#F18F01',   # Naranja
    'residual': '#C73E1D',  # Rojo
    'grid': '#E8E8E8'
}

# =========================
# GRÁFICOS INDIVIDUALES
# =========================

def plot_density(y_true, y_pred, ax, title=None, xlabel="Y"):
    """Gráfico de densidad con KDE suavizado"""
    
    # Histogramas con transparencia
    n_bins = min(30, len(y_true) // 10)
    
    ax.hist(y_true, bins=n_bins, alpha=0.3, density=True, 
            color=COLORS['real'], edgecolor='white', linewidth=0.5,
            label='Real')
    ax.hist(y_pred, bins=n_bins, alpha=0.3, density=True,
            color=COLORS['pred'], edgecolor='white', linewidth=0.5,
            label='Predicho')
    
    # KDE suavizado
    try:
        kde_true = stats.gaussian_kde(y_true)
        kde_pred = stats.gaussian_kde(y_pred)
        
        x_range = np.linspace(
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
            200
        )
        
        ax.plot(x_range, kde_true(x_range), 
                color=COLORS['real'], linewidth=2.5, alpha=0.8)
        ax.plot(x_range, kde_pred(x_range),
                color=COLORS['pred'], linewidth=2.5, alpha=0.8)
    except:
        pass  # Si falla KDE, solo mostrar histogramas
    
    # Estilo
    ax.set_xlabel(xlabel, fontsize=11, fontweight='semibold')
    ax.set_ylabel('Densidad', fontsize=11, fontweight='semibold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Leyenda elegante
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                      fontsize=10, loc='best')
    legend.get_frame().set_alpha(0.9)


def plot_scatter(y_true, y_pred, ax, title=None):
    """Scatter plot con métricas y línea de tendencia"""
    
    # Métricas
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Scatter con gradiente de densidad
    ax.scatter(y_true, y_pred, alpha=0.4, s=25, 
              c=COLORS['pred'], edgecolors='white', linewidth=0.5)
    
    # Línea perfecta
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    ax.plot(lims, lims, color=COLORS['perfect'], linestyle='--', 
            linewidth=2.5, label='Predicción perfecta', alpha=0.8)
    
    # Línea de regresión real
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(lims, p(lims), color=COLORS['real'], linestyle='-',
            linewidth=2, label=f'Tendencia (y={z[0]:.2f}x+{z[1]:.2f})', alpha=0.7)
    
    # Estilo
    ax.set_xlabel('Valor Real', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Valor Predicho', fontsize=11, fontweight='semibold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Título con métricas
    if title:
        title_text = f"{title}\n"
    else:
        title_text = ""
    
    title_text += f"R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}"
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=10)
    
    # Leyenda
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      fontsize=9, loc='best')
    legend.get_frame().set_alpha(0.9)
    
    # Aspecto cuadrado para mejor visualización
    ax.set_aspect('equal', adjustable='box')


def plot_residuals(y_true, y_pred, ax, title=None):
    """Gráfico de residuos con bandas de confianza"""
    
    residuals = y_true - y_pred
    
    # Scatter de residuos
    ax.scatter(y_pred, residuals, alpha=0.4, s=25,
              c=COLORS['residual'], edgecolors='white', linewidth=0.5)
    
    # Línea de referencia en 0
    ax.axhline(0, color='black', linestyle='--', linewidth=2, 
               label='Residuo = 0', alpha=0.7)
    
    # Bandas de confianza (±2 desviaciones estándar)
    std_res = np.std(residuals)
    ax.axhline(2 * std_res, color=COLORS['perfect'], linestyle=':', 
               linewidth=1.5, alpha=0.6, label='±2σ')
    ax.axhline(-2 * std_res, color=COLORS['perfect'], linestyle=':', 
               linewidth=1.5, alpha=0.6)
    
    # Línea de tendencia LOWESS (suavizada)
    try:
        from scipy.signal import savgol_filter
        sorted_idx = np.argsort(y_pred)
        y_sorted = y_pred[sorted_idx]
        res_sorted = residuals[sorted_idx]
        
        if len(res_sorted) > 10:
            window = min(51, len(res_sorted) // 3)
            if window % 2 == 0:
                window += 1
            smooth = savgol_filter(res_sorted, window, 3)
            ax.plot(y_sorted, smooth, color=COLORS['real'], 
                   linewidth=2.5, label='Tendencia', alpha=0.8)
    except:
        pass
    
    # Estilo
    ax.set_xlabel('Valor Predicho', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Residuo (Real - Predicho)', fontsize=11, fontweight='semibold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Título con estadísticas de residuos
    if title:
        title_text = f"{title}\n"
    else:
        title_text = ""
    
    title_text += f"Media = {np.mean(residuals):.4f} | σ = {std_res:.4f}"
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=10)
    
    # Leyenda
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      fontsize=9, loc='best')
    legend.get_frame().set_alpha(0.9)


def plot_residuals_histogram(residuals, ax, title=None):
    """Histograma de residuos con curva normal"""
    
    # Histograma
    n, bins, patches = ax.hist(residuals, bins=30, density=True, 
                               alpha=0.6, color=COLORS['residual'],
                               edgecolor='white', linewidth=0.5,
                               label='Residuos')
    
    # Curva normal teórica
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma),
            color=COLORS['real'], linewidth=2.5, 
            label=f'N({mu:.2f}, {sigma:.2f}²)', alpha=0.8)
    
    # Estilo
    ax.set_xlabel('Residuo', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Densidad', fontsize=11, fontweight='semibold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      fontsize=10, loc='best')
    legend.get_frame().set_alpha(0.9)


# =========================
# PIPELINE DE ANÁLISIS
# =========================

def plot_regression_analysis(
    splits,
    output_path,
    model_name="Modelo"
):
    """
    splits: lista de tuplas (y_true, y_pred, label)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_splits = len(splits)

    # =========================
    # DENSIDADES + SCATTER
    # =========================
    fig = plt.figure(figsize=(6 * n_splits, 10))
    fig.suptitle(
        f"Análisis de Regresión - {model_name}",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    
    gs = fig.add_gridspec(2, n_splits, hspace=0.3, wspace=0.3)

    for i, (y_true, y_pred, label) in enumerate(splits):
        ax_density = fig.add_subplot(gs[0, i])
        ax_scatter = fig.add_subplot(gs[1, i])
        
        plot_density(
            y_true,
            y_pred,
            ax=ax_density,
            title=f"Distribución - {label}"
        )
        plot_scatter(
            y_true,
            y_pred,
            ax=ax_scatter,
            title=label
        )

    plt.savefig(
        output_path / "analisis_regresion.png",
        dpi=300,
        bbox_inches="tight",
        facecolor='white',
        edgecolor='none'
    )
    plt.close()

    # =========================
    # RESIDUOS + HISTOGRAMAS
    # =========================
    fig = plt.figure(figsize=(6 * n_splits, 10))
    fig.suptitle(
        f"Análisis de Residuos - {model_name}",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )
    
    gs = fig.add_gridspec(2, n_splits, hspace=0.3, wspace=0.3)

    for i, (y_true, y_pred, label) in enumerate(splits):
        ax_scatter = fig.add_subplot(gs[0, i])
        ax_hist = fig.add_subplot(gs[1, i])
        
        plot_residuals(
            y_true,
            y_pred,
            ax=ax_scatter,
            title=f"Residuos - {label}"
        )
        
        residuals = y_true - y_pred
        plot_residuals_histogram(
            residuals,
            ax=ax_hist,
            title=f"Distribución de Residuos - {label}"
        )

    plt.savefig(
        output_path / "residuos_regresion.png",
        dpi=300,
        bbox_inches="tight",
        facecolor='white',
        edgecolor='none'
    )
    plt.close()
