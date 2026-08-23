"""
viz_evaluacion.py
=================
Figuras de la evaluación fuera de muestra que los notebooks 11_04 y siguientes
consumen. Tres vistas, una por pregunta:

plot_ventana_movil     ¿cómo evoluciona el error a lo largo del tiempo y qué
                       pasa al cruzar T0? Entrenamiento y prueba en el MISMO
                       eje, con el corte marcado.
plot_bandas_serie      ¿la banda de credibilidad sigue a la serie del score, y
                       dónde falla? Cubre train y test, no sólo test.
plot_extractos_curvas  ¿cómo se ve el intervalo sobre la curva misma? Extractos
                       de la serie funcional cada `cada` períodos, cada uno con
                       su banda y la curva verdadera encima.

Convención de color, común a las tres: el bloque de entrenamiento va en azul
frío y el de prueba en rojo. La distinción es la lectura principal de todas
estas figuras y por eso no se deja al orden de la paleta por defecto.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_ventana_movil",
    "plot_bandas_serie",
    "plot_extractos_curvas",
    "plot_calibracion_pit",
]

C_TRAIN = "#1a6faf"
C_TEST = "#c0392b"
C_OBS = "0.30"


def _guardar(fig, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _sombrear_bloques(ax, T0: int, t_min: float, t_max: float) -> None:
    """Fondo tenue que separa entrenamiento de prueba y marca el corte en T0."""
    ax.axvspan(t_min, T0, color=C_TRAIN, alpha=0.05, lw=0, zorder=0)
    ax.axvspan(T0, t_max, color=C_TEST, alpha=0.05, lw=0, zorder=0)
    ax.axvline(T0, color="k", lw=1.2, ls="--", alpha=0.8, zorder=1)


# ==========================================================================
# 1. VENTANA MÓVIL
# ==========================================================================

def plot_ventana_movil(tabla: pd.DataFrame, T0: int, metricas: Sequence[str],
                       columna_grupo: Optional[str] = None,
                       tablas_por_w: Optional[Dict[int, pd.DataFrame]] = None,
                       title: str = "Evolución del error sobre ventana móvil",
                       save_path: Optional[str] = None):
    """
    Una fila por métrica; el eje x es el tiempo del experimento.

    tabla : salida de `fit.ventana_movil_scores` o `ventana_movil_funcional`.
    metricas : columnas a dibujar, una por panel.
    columna_grupo : si se indica (p. ej. "componente"), dibuja una línea por
        grupo dentro de cada panel. Para la versión funcional se deja en None.
    tablas_por_w : {w: tabla} para superponer varios anchos de ventana. Sirve
        para mostrar que la conclusión no depende del w elegido a dedo; el w
        mayor va más opaco.

    Las ventanas que CRUZAN T0 se dibujan punteadas: su cifra mezcla dentro y
    fuera de muestra y no debe leerse como ninguno de los dos.
    """
    metricas = list(metricas)
    fig, axes = plt.subplots(len(metricas), 1, figsize=(12, 2.9 * len(metricas)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    fuentes = tablas_por_w if tablas_por_w else {tabla.attrs.get("w", 0): tabla}
    ws = sorted(fuentes)

    for ax, met in zip(axes, metricas):
        t_min = min(float(t["t_centro"].min()) for t in fuentes.values())
        t_max = max(float(t["t_centro"].max()) for t in fuentes.values())
        _sombrear_bloques(ax, T0, t_min, t_max)

        for i_w, w in enumerate(ws):
            tw = fuentes[w]
            if met not in tw.columns:
                continue
            alpha = 0.35 + 0.65 * (i_w + 1) / len(ws)
            grupos = ([(g, tw[tw[columna_grupo] == g])
                       for g in tw[columna_grupo].unique()]
                      if columna_grupo else [(None, tw)])
            for i_g, (nombre, sub) in enumerate(grupos):
                sub = sub.sort_values("t_centro")
                color = (plt.cm.tab10(i_g % 10) if columna_grupo
                         else (C_TRAIN if len(ws) == 1 else plt.cm.viridis(i_w / max(len(ws) - 1, 1))))
                etiqueta = " · ".join(
                    [s for s in [str(nombre) if nombre else "",
                                 f"w={w}" if len(ws) > 1 else ""] if s]) or None
                ax.plot(sub["t_centro"], sub[met], color=color, lw=1.5,
                        alpha=alpha, label=etiqueta)
                # tramo que cruza el corte: mismo color, punteado
                cruza = sub[sub["cruza_T0"]]
                if not cruza.empty:
                    ax.plot(cruza["t_centro"], cruza[met], color=color,
                            lw=1.5, ls=":", alpha=alpha)

        ax.set_ylabel(met)
        if met in ("cobertura", "cobertura_puntual"):
            ax.axhline(0.95, color="k", lw=0.9, ls="-.", alpha=0.6)
            ax.set_ylim(0, 1.02)
        if met == "r2_local":
            ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        if ax is axes[0]:
            ax.text(T0, ax.get_ylim()[1], "  $T_0$", va="top", ha="left",
                    fontsize=9, color="k")
            h, l = ax.get_legend_handles_labels()
            if l:
                ax.legend(fontsize=8, ncol=min(len(l), 5), loc="upper left")

    axes[-1].set_xlabel("tiempo $t$ (centro de la ventana)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _guardar(fig, save_path)
    return fig


# ==========================================================================
# 2. BANDAS SOBRE LA SERIE DE SCORES
# ==========================================================================

def plot_bandas_serie(y_obs: np.ndarray, y_hat: np.ndarray,
                      li: np.ndarray, ls: np.ndarray, T0: int,
                      t: Optional[np.ndarray] = None,
                      etiquetas: Optional[Sequence[str]] = None,
                      nivel: float = 0.95,
                      title: str = "Bandas de credibilidad por score",
                      save_path: Optional[str] = None):
    """
    Serie de cada score con su banda, cubriendo entrenamiento y prueba.

    y_obs, y_hat, li, ls : (n, M). `t` es el tiempo del experimento (base-1);
    si se omite se asume 1..n. Los puntos fuera de banda se marcan y se cuentan
    por separado en cada bloque, que es la comparación que interesa: una banda
    puede estar perfectamente calibrada dentro de muestra y fallar fuera.
    """
    Y = np.atleast_2d(y_obs); P = np.atleast_2d(y_hat)
    L = np.atleast_2d(li);    U = np.atleast_2d(ls)
    n, M = Y.shape
    t = np.arange(1, n + 1) if t is None else np.asarray(t).ravel()
    etiquetas = list(etiquetas) if etiquetas else [f"FPC {m+1}" for m in range(M)]

    fig, axes = plt.subplots(M, 1, figsize=(13, 2.7 * M), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for m, ax in enumerate(axes):
        _sombrear_bloques(ax, T0, float(t[0]), float(t[-1]))
        ax.fill_between(t, L[:, m], U[:, m], color=C_TEST, alpha=0.20, lw=0,
                        label=f"banda {int(nivel*100)}% (cuantiles muestrales)")
        ax.plot(t, Y[:, m], color=C_OBS, lw=0.9, label=r"$\xi$ observado")
        ax.plot(t, P[:, m], color=C_TEST, lw=1.1, label="media predictiva")

        fuera = (Y[:, m] < L[:, m]) | (Y[:, m] > U[:, m])
        ax.scatter(t[fuera], Y[fuera, m], s=14, color="k", zorder=3)

        es_train = t <= T0
        c_tr = float(np.mean(~fuera[es_train])) if es_train.any() else np.nan
        c_te = float(np.mean(~fuera[~es_train])) if (~es_train).any() else np.nan
        ax.set_title(f"{etiquetas[m]} — cobertura train={c_tr:.3f} · "
                     f"test={c_te:.3f}  (nominal {nivel:.2f}) · "
                     f"{int(fuera.sum())} fuera de banda",
                     fontsize=10, loc="left")
        ax.set_ylabel(etiquetas[m])
        if m == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper right")

    axes[-1].set_xlabel("tiempo $t$")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _guardar(fig, save_path)
    return fig


# ==========================================================================
# 3. EXTRACTOS DE LA SERIE FUNCIONAL CON SU BANDA
# ==========================================================================

def plot_extractos_curvas(X_true: np.ndarray, X_pred: np.ndarray,
                          li: np.ndarray, ls: np.ndarray,
                          tau: np.ndarray, T0: int,
                          t: Optional[np.ndarray] = None,
                          cada: int = 10, n_col: int = 5,
                          nivel: float = 0.95,
                          X_obs: Optional[np.ndarray] = None,
                          title: str = "Intervalos de credibilidad sobre la curva",
                          save_path: Optional[str] = None):
    """
    Rejilla de curvas tomadas cada `cada` períodos, cada una con su banda.

    X_true : (n, G) curva VERDADERA del generador, que es contra lo que se
        evalúa. `X_obs` es opcional y se dibuja como puntos tenues para mostrar
        dónde estaban los datos ruidosos; no es el objetivo de la predicción.
    li, ls : (n, G) banda PUNTUAL de nivel `nivel`. No es simultánea: cubre
        cada tau por separado, de modo que la probabilidad de contener la curva
        entera es menor que el nominal.

    El título de cada panel indica el bloque y la fracción del dominio cubierta
    por la banda en ese período, que es lo que permite ver de un vistazo si el
    modelo falla en curvas concretas y no de forma difusa.
    """
    Xt = np.atleast_2d(X_true); Xp = np.atleast_2d(X_pred)
    L = np.atleast_2d(li);      U = np.atleast_2d(ls)
    n, G = Xt.shape
    tau = np.asarray(tau, dtype=float).ravel()
    t = np.arange(1, n + 1) if t is None else np.asarray(t).ravel()

    idx = np.arange(0, n, int(cada))
    n_fil = int(np.ceil(len(idx) / n_col))
    fig, axes = plt.subplots(n_fil, n_col, figsize=(3.0 * n_col, 2.7 * n_fil),
                             sharex=True, sharey=True, squeeze=False)
    planos = axes.ravel()

    for ax, i in zip(planos, idx):
        es_test = t[i] > T0
        color = C_TEST if es_test else C_TRAIN
        ax.fill_between(tau, L[i], U[i], color=color, alpha=0.22, lw=0)
        if X_obs is not None:
            ax.plot(tau, np.atleast_2d(X_obs)[i], ".", color="0.65", ms=2.2,
                    alpha=0.8)
        ax.plot(tau, Xt[i], color=C_OBS, lw=1.3)
        ax.plot(tau, Xp[i], color=color, lw=1.4)

        cob = float(np.mean((Xt[i] >= L[i]) & (Xt[i] <= U[i])))
        ax.set_title(rf"$X_{{{int(t[i])}}}$ · {'test' if es_test else 'train'}"
                     f" · cob={cob:.2f}", fontsize=9, color=color)
        ax.tick_params(labelsize=7)

    for ax in planos[len(idx):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\tau$", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$X_t(\tau)$", fontsize=9)

    manijas = [
        plt.Line2D([], [], color=C_OBS, lw=1.3, label="curva verdadera"),
        plt.Line2D([], [], color=C_TRAIN, lw=1.4, label="predicción (train)"),
        plt.Line2D([], [], color=C_TEST, lw=1.4, label="predicción (test)"),
        plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=0.25,
                      label=f"banda {int(nivel*100)}% puntual"),
    ]
    if X_obs is not None:
        manijas.append(plt.Line2D([], [], color="0.65", marker=".", ls="",
                                  label="datos observados (con ruido)"))
    fig.legend(handles=manijas, loc="lower center", ncol=len(manijas),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{title} — un extracto cada {cada} períodos", fontsize=12)
    fig.tight_layout()
    _guardar(fig, save_path)
    return fig


# ==========================================================================
# 4. CALIBRACIÓN: PIT POR BLOQUE
# ==========================================================================

def plot_calibracion_pit(pit_train: Dict[str, np.ndarray],
                         pit_test: Dict[str, np.ndarray],
                         n_bins: int = 10,
                         title: str = "Calibración: histograma PIT por bloque",
                         save_path: Optional[str] = None):
    """
    Histograma PIT por componente, entrenamiento contra prueba.

    Bajo calibración perfecta el PIT es uniforme. La forma dice qué falla:
    una U indica bandas demasiado angostas (sub-dispersión), una campana
    demasiado anchas, y una pendiente un sesgo sistemático del centro.
    Contrastar train y test separa el desajuste del modelo de la pérdida de
    calibración fuera de muestra, que tienen causas distintas.
    """
    claves = list(pit_train)
    fig, axes = plt.subplots(1, len(claves), figsize=(3.4 * len(claves), 3.1),
                             squeeze=False, sharey=True)
    bordes = np.linspace(0, 1, n_bins + 1)

    for ax, k in zip(axes[0], claves):
        for u, color, etq in ((pit_train[k], C_TRAIN, "train"),
                              (pit_test.get(k), C_TEST, "test")):
            if u is None:
                continue
            ax.hist(u, bins=bordes, density=True, histtype="step", lw=1.8,
                    color=color, label=etq)
        ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(k, fontsize=10)
        ax.set_xlabel("PIT")
        ax.set_xlim(0, 1)
    axes[0, 0].set_ylabel("densidad")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _guardar(fig, save_path)
    return fig
