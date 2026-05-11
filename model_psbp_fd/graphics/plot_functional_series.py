def plot_functional_time_series(X, grid, *, highlight_idx=None,
                                color="#1a6faf", title=None,
                                separator_every=1, figsize=(14, 4), dpi=130):
    """
    Serie de tiempo funcional como línea continua.

    La curva X_t(s) ocupa el intervalo [t-1, t] del eje horizontal.
    Se lee igual que una serie escalar: de izquierda a derecha en el tiempo.
    Las líneas verticales grises separan cada observación funcional.

    Parámetros
    ----------
    X              : (T, G)  — serie simulada.
    grid           : (G,)    — puntos del dominio s.
    highlight_idx  : índices t (0-based) a etiquetar. Default: 0, 1, T//2, T-1.
    color          : color de la línea.
    title          : título (auto si None).
    separator_every: cada cuántos t dibujar separador vertical.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    T, G = X.shape
    highlight_idx = highlight_idx or [0, 1, T // 2, T - 1]

    # Una sola línea continua: curva t se desplaza t unidades en x
    x_global = np.concatenate([t + grid for t in range(T)])
    y_global = X.ravel()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(x_global, y_global, color=color, lw=0.85, alpha=0.92)

    # Separadores entre observaciones funcionales
    for t in range(1, T, separator_every):
        ax.axvline(t, color="#cccccc", lw=0.6, zorder=0)

    # Etiquetas sobre curvas seleccionadas
    y_range = X.max() - X.min()
    for t in highlight_idx:
        x_mid = t + 0.5
        y_top = X[t].max()
        label = (r"$X_1(s)$"        if t == 0
                 else r"$X_T(s)$"   if t == T - 1
                 else fr"$X_{{{t+1}}}(s)$")
        ax.annotate(label, xy=(x_mid, y_top),
                    xytext=(x_mid, y_top + 0.18 * y_range),
                    ha="center", fontsize=8.5, color=color,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.7))

    tick_step = max(1, T // 10)
    ax.set_xticks(range(0, T + 1, tick_step))
    ax.set_xticklabels([str(v) for v in range(0, T + 1, tick_step)], fontsize=8)
    ax.set_xlim(0, T)
    ax.set_xlabel(r"Tiempo $t$  (intervalo $[t-1,\,t]$ = curva $X_t(s)$)", fontsize=9)
    ax.set_ylabel(r"$X_t(s)$", fontsize=10)
    ax.set_title(title or f"Serie de tiempo funcional  —  $T={T}$ observaciones",
                 fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    plt.tight_layout()
    return fig
