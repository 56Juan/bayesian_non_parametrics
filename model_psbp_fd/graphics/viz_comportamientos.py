"""
viz_comportamientos.py
======================
Figuras de los predictores puntuales (docs 03_05_04) y de los comportamientos
predictivos (03_06). Solo dibujan: las cifras las calculan
`fit/resumenes_predictiva.py`, `fit/evaluacion_barrido.py` y
`fit/comportamientos.py`.

Funciones publicas
------------------
ESTILOS_PREDICTORES        : color y trazo de cada predictor.
plot_predictores_vs_M      : Bloque A en test contra M, una linea por predictor.
plot_histogramas_draws     : extracciones de cada score en un origen, con esperanza,
                             mediana, modal y observado.
plot_curvas_predictores    : extracciones funcionales con los cinco predictores.
plot_comportamientos       : curvas tipicas y bandas condicionales por origen.
plot_K_barrido             : distribucion de K por M.
plot_K_serie               : K por origen a lo largo del tiempo.
plot_catalogo              : comportamientos globales y su probabilidad por origen.
plot_validacion_regimen    : cruce con el regimen verdadero contra M.
plot_validacion_observado  : las tres cifras de la validacion contra lo observado vs M.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "ESTILOS_PREDICTORES", "plot_predictores_vs_M", "plot_histogramas_draws",
    "plot_curvas_predictores", "plot_comportamientos", "plot_K_barrido", "plot_K_serie",
    "plot_catalogo", "plot_validacion_regimen", "plot_validacion_observado",
]

ESTILOS_PREDICTORES = {
    "esperanza": dict(color="#c0392b", lw=1.8, ls="-"),
    "mediana":   dict(color="#2c7fb8", lw=1.5, ls="--"),
    "medoide":   dict(color="#2ca25f", lw=1.5, ls="-."),
    "mbd":       dict(color="#8856a7", lw=1.5, ls=":"),
    "modal":     dict(color="#e08e0b", lw=1.5, ls=(0, (5, 1, 1, 1))),
}
_PALETA = plt.get_cmap("tab10").colors


def _guardar(fig, save_path):
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def _rejilla(n: int, ncols: int, ancho: float, alto: float, **kw):
    ncols = min(n, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ancho * ncols, alto * nrows + 0.8), **kw)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    return fig, [axes[j // ncols, j % ncols] for j in range(n)]


def plot_predictores_vs_M(tabla: pd.DataFrame, metricas, titulo: str, save_path=None) -> plt.Figure:
    """`tabla` de `tabla_predictores` (filas (columna, predictor), columnas M);
    `metricas` como METRICAS_A. Un panel por metrica, una linea por predictor."""
    fig, axes = plt.subplots(1, len(metricas), figsize=(4.2 * len(metricas), 3.6), squeeze=False)
    for ax, (etiqueta, col) in zip(axes[0], metricas):
        sub = tabla.loc[col]
        for p in sub.index:
            est = ESTILOS_PREDICTORES.get(p, {})
            ax.plot(sub.columns, sub.loc[p], marker="o", ms=4, label=p, **est)
        ax.set_xticks(list(sub.columns))
        ax.set_xlabel("M")
        ax.set_title(etiqueta, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(titulo, fontsize=11)
    return _guardar(fig, save_path)


def plot_histogramas_draws(SC: np.ndarray, i: int, marcas: Dict[str, np.ndarray], t: int,
                           M: int, q=(0.005, 0.995), ncols: int = 5, bins: int = 60,
                           save_path=None) -> plt.Figure:
    """
    Extracciones de cada score en el origen `i` (`SC` (S, n, M)), recortadas a los
    cuantiles `q` solo para el dibujo, con lineas verticales en `marcas`
    {nombre: (n, M)} (esperanza, mediana, modal, observado...).
    """
    n_k = SC.shape[2]
    fig, axes = _rejilla(n_k, ncols, 3.0, 2.4)
    estilos = {**ESTILOS_PREDICTORES, "observado": dict(color="k", lw=1.8, ls="-")}
    for k, ax in enumerate(axes):
        z = np.asarray(SC[:, i, k], float)
        lo, hi = np.quantile(z, q)
        ax.hist(z[(z >= lo) & (z <= hi)], bins=bins, color="0.75", density=True)
        for nombre, v in marcas.items():
            est = estilos.get(nombre, {})
            ax.axvline(float(v[i, k]), color=est.get("color", "k"), ls=est.get("ls", "-"),
                       lw=est.get("lw", 1.4), label=nombre)
        ax.set_title(rf"$\xi_{{{k + 1}}}$", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"M={M} · t={t} · extracciones de cada score (recortadas a "
                 f"{q[0]:.1%}–{q[1]:.1%} para el dibujo)", fontsize=11)
    return _guardar(fig, save_path)


def plot_curvas_predictores(Xd: np.ndarray, preds: Dict[str, np.ndarray], X_obj: np.ndarray,
                            grilla: np.ndarray, t: Sequence[int], M: int, n_curvas: int = 150,
                            ncols: int = 4, seed: int = 0, save_path=None) -> plt.Figure:
    """Por origen (columnas de `Xd` (S, n', G)): `n_curvas` extracciones en gris,
    los predictores `preds` {nombre: (n', G)} y la curva objetivo en negro."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(Xd.shape[0], size=min(n_curvas, Xd.shape[0]), replace=False)
    fig, axes = _rejilla(len(t), ncols, 3.6, 3.0, sharey=True)
    for j, ax in enumerate(axes):
        ax.plot(grilla, np.asarray(Xd[sel, j]).T, color="0.6", lw=0.4, alpha=0.35)
        for nombre, X in preds.items():
            ax.plot(grilla, X[j], label=nombre, **ESTILOS_PREDICTORES.get(nombre, {}))
        ax.plot(grilla, X_obj[j], color="k", lw=1.8, label="objetivo")
        ax.set_title(f"t={int(t[j])}", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"M={M} · extracciones de la predictiva funcional y los predictores puntuales",
                 fontsize=11)
    return _guardar(fig, save_path)


def plot_comportamientos(C: Dict, idx: Sequence[int], X_obj: np.ndarray, li_f: np.ndarray,
                         ls_f: np.ndarray, grilla: np.ndarray, t_orig: np.ndarray, M: int,
                         X_esp: Optional[np.ndarray] = None, ncols: int = 4,
                         save_path=None) -> plt.Figure:
    """
    Por origen de `idx`: banda marginal (gris), y por comportamiento su banda
    condicional y su curva tipica con la masa en la leyenda; la esperanza
    (opcional) punteada y la curva objetivo en negro.
    """
    fig, axes = _rejilla(len(idx), ncols, 3.8, 3.1, sharey=True)
    for ax, i in zip(axes, idx):
        ax.fill_between(grilla, li_f[i], ls_f[i], color="0.85", lw=0, label="banda marginal")
        for c in range(int(C["K"][i])):
            col = _PALETA[c % len(_PALETA)]
            ax.fill_between(grilla, C["li_c"][i, c], C["ls_c"][i, c], color=col, alpha=0.18, lw=0)
            ax.plot(grilla, C["curvas"][i, c], color=col, lw=1.6, label=f"c{c + 1}: p={C['masas'][i, c]:.2f}")
        if X_esp is not None:
            ax.plot(grilla, X_esp[i], color="#c0392b", lw=1.1, ls=":", label="esperanza")
        ax.plot(grilla, X_obj[i], color="k", lw=1.6, label="objetivo")
        ax.set_title(f"t={int(t_orig[i])} · K={int(C['K'][i])}", fontsize=9)
        ax.legend(fontsize=6.5, loc="best")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"M={M} · comportamientos predictivos: curva tipica y banda condicional "
                 f"(nivel {C['nivel']:.0%}) por grupo", fontsize=11)
    return _guardar(fig, save_path)


def plot_K_barrido(tabla: pd.DataFrame, titulo: str, save_path=None) -> plt.Figure:
    """Barras apiladas: fraccion de origenes con cada K, por M, en train y test
    (`102_K_por_M.csv`)."""
    bloques = [b for b in ("train", "test") if b in set(tabla["bloque"])]
    fig, axes = plt.subplots(1, len(bloques), figsize=(6.0 * len(bloques), 3.6), squeeze=False, sharey=True)
    for ax, b in zip(axes[0], bloques):
        piv = tabla[tabla.bloque == b].pivot(index="M", columns="K", values="frac").fillna(0.0)
        piv = piv.loc[:, piv.sum(axis=0) > 0]
        base = np.zeros(len(piv))
        for j, K in enumerate(piv.columns):
            ax.bar(piv.index.astype(str), piv[K], bottom=base, color=_PALETA[j % len(_PALETA)], label=f"K={K}")
            base += piv[K].to_numpy()
        ax.set_xlabel("M")
        ax.set_title(f"{b}", fontsize=10)
        ax.set_ylim(0, 1.0)
    axes[0, 0].set_ylabel("fraccion de origenes")
    axes[0, -1].legend(fontsize=8, loc="lower right")
    fig.suptitle(titulo, fontsize=11)
    return _guardar(fig, save_path)


def plot_K_serie(K_por_M: Dict[int, np.ndarray], t: np.ndarray, T0: int,
                 n_B: Optional[np.ndarray] = None, titulo: str = "", save_path=None) -> plt.Figure:
    """K por origen en el tiempo, una fila por M; con `n_B` (n,), una fila extra
    con el numero de scores activos en el regimen B (verdad del generador)."""
    filas = list(K_por_M) + (["regimen"] if n_B is not None else [])
    fig, axes = plt.subplots(len(filas), 1, figsize=(12, 0.9 * len(filas) + 1.2), sharex=True, squeeze=False)
    for ax, M in zip(axes[:, 0], filas):
        v = n_B if M == "regimen" else K_por_M[M]
        ok = v > 0 if M != "regimen" else np.ones(len(v), bool)
        ax.scatter(t[ok], v[ok], s=3, c=v[ok], cmap="viridis", vmin=0, vmax=max(3, float(np.max(v))))
        ax.axvline(T0, color="crimson", ls="--", lw=1)
        ax.set_ylabel("n en B" if M == "regimen" else f"K (M={M})", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("t")
    fig.suptitle(titulo, fontsize=11)
    return _guardar(fig, save_path)


def plot_catalogo(cat: Dict, grilla: np.ndarray, t: np.ndarray, M: int, save_path=None) -> plt.Figure:
    """Izquierda: curva de cada comportamiento global con su masa. Derecha: su
    probabilidad por origen (solo los origenes del catalogo)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 2.2]})
    for c in range(cat["K"]):
        a1.plot(grilla, cat["curvas"][c], color=_PALETA[c % len(_PALETA)], lw=1.8,
                label=f"G{c + 1}: masa {cat['masas'][c]:.2f}")
    a1.legend(fontsize=8)
    a1.set_xlabel(r"$\tau$")
    a1.set_title("comportamientos globales", fontsize=10)
    ok = np.isfinite(cat["prob"][:, 0])
    a2.stackplot(t[ok], cat["prob"][ok].T, colors=[_PALETA[c % len(_PALETA)] for c in range(cat["K"])])
    a2.set_ylim(0, 1)
    a2.set_xlabel("t")
    a2.set_title("probabilidad de cada comportamiento por origen", fontsize=10)
    fig.suptitle(f"M={M} · catalogo global (L={cat['K']})", fontsize=11)
    return _guardar(fig, save_path)


def plot_validacion_regimen(tabla: pd.DataFrame, titulo: str, save_path=None) -> plt.Figure:
    """`104_validacion_regimen_por_M.csv` contra M: acuerdo en K y fracciones con
    K >= 2, TV y pureza, aciertos del modelo y del oraculo."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    M = tabla["M"]
    axes[0].plot(M, tabla["acuerdo_K"], "o-", color="k", label="acuerdo K modelo = K oraculo")
    axes[0].plot(M, tabla["frac_K2mas_modelo"], "s--", color="#c0392b", label="K>=2 modelo")
    axes[0].plot(M, tabla["frac_K2mas_oraculo"], "^--", color="#2c7fb8", label="K>=2 oraculo")
    axes[1].plot(M, tabla["tv"], "o-", color="#8856a7", label="TV(p modelo, q oraculo)")
    axes[1].plot(M, tabla["pureza"], "s-", color="#2ca25f", label="pureza de regimen")
    axes[2].plot(M, tabla["acierto_modelo"], "o-", color="#c0392b", label="acierto modelo")
    axes[2].plot(M, tabla["acierto_oraculo"], "s--", color="#2c7fb8", label="acierto oraculo")
    for ax in axes:
        ax.set_xticks(list(M))
        ax.set_xlabel("M")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    fig.suptitle(titulo, fontsize=11)
    return _guardar(fig, save_path)


def plot_validacion_observado(tabla: pd.DataFrame, objetivo: str, subconjunto: str,
                              titulo: str, save_path=None) -> plt.Figure:
    """`105_validacion_comportamientos_por_M.csv` en test: p del comportamiento
    observado contra la esperada y acierto; PICP y MPIW de las bandas condicional
    del realizado, del mas probable y marginal."""
    d = tabla[(tabla.objetivo == objetivo) & (tabla.subconjunto == subconjunto)
              & (tabla.bloque == "test")].sort_values("M")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    M = d["M"]
    axes[0].plot(M, d["p_obs"], "o-", color="k", label="p del comportamiento observado")
    axes[0].plot(M, d["p_esperada"], "s--", color="0.5", label="esperada (sum p^2)")
    axes[0].plot(M, d["acierto"], "^-", color="#c0392b", label="acierto del mas probable")
    for s, est in (("real", dict(color="#2ca25f", marker="o")), ("mp", dict(color="#e08e0b", marker="s")),
                   ("marg", dict(color="#2c7fb8", marker="^"))):
        axes[1].plot(M, d[f"picp_{s}"], label=s, **est)
        axes[2].plot(M, d[f"mpiw_{s}"], label=s, **est)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_title("PICP", fontsize=10)
    axes[2].set_title("MPIW", fontsize=10)
    for ax in axes:
        ax.set_xticks(list(M))
        ax.set_xlabel("M")
        ax.legend(fontsize=7)
    fig.suptitle(titulo, fontsize=11)
    return _guardar(fig, save_path)
