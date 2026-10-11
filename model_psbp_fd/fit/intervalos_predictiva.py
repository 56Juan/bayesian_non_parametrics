"""
intervalos_predictiva.py
========================
Las tres construcciones de intervalo de credibilidad predictivo de docs/03 Modelo.tex
§03_05_05, sobre extracciones funcionales `Xd` (S, n, G) de la predictiva del
PSBPM-FD, y las metricas del Bloque B sobre una UNION DE TRAMOS por punto.

    equicolas   cuantiles zeta/2 y 1 - zeta/2 de la muestra en cada tau_g
                (eq. intervalo_puntual). Es la banda de siempre: `_predecir_punto`
                la calcula aqui, con las mismas operaciones, y la cifra es identica.
    simultanea  banda de tipo supremo (eq. supnorma, Degras 2011): centro c y escala
                s puntuales, D^(s) = max_g |X^(s) - c| / s, q = cuantil `nivel` de D,
                banda c +- q s. Centro = mediana y escala = IQR / 1.349, NO media y sd:
                con atau <= 1 las extracciones tienen colas no acotadas y la sd la
                arrastran unos pocos draws (CLAUDE.md §8.7). Contiene enteras a una
                fraccion `nivel` de las extracciones.
    hdr         region de maxima densidad PUNTUAL en tau (eq. hdr, Hyndman 1996),
                no una HDR conjunta sobre la curva. Ver `hdr_puntual`.

Las metricas por tramos reciben (L, U) de forma (n, G, K), con NaN donde no hay
tramo (la HDR de un punto unimodal es un solo tramo), o (n, G) para una banda de un
tramo. Con K = 1 reproducen al bit `metrics_distribucional.winkler`, `mpiw`,
`indicador_cobertura` e `indicador_cobertura_simultanea`:

    winkler_tramos  sum_k (U_k - L_k) + (2 / (1 - nivel)) dist(y, U_k [L_k, U_k])
    ancho_tramos    sum_k (U_k - L_k) por punto; `mpiw_tramos` su promedio
    cobertura_tramos             1{y en algun tramo}
    cobertura_simultanea_tramos  todos los tau_g de la curva cubiertos
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..utils.quadrature import integrar, pesos_trapezoidales

__all__ = [
    "INTERVALOS", "K_MAX_HDR", "RECORTE_HDR", "N_GRID_HDR",
    "banda_equicolas", "banda_simultanea", "hdr_puntual",
    "winkler_tramos", "ancho_tramos", "mpiw_tramos", "cobertura_tramos",
    "cobertura_simultanea_tramos", "resumen_intervalo_tramos",
]

#: Las tres construcciones de docs §03_05_05, en el orden en que se reportan.
INTERVALOS = ("equicolas", "simultanea", "hdr")
K_MAX_HDR = 3                    # tramos por punto; con mas se fusionan los mas cercanos
RECORTE_HDR = (0.005, 0.995)     # grilla de la HDR cuando debe_recortar(hp_json)
N_GRID_HDR = 256


# ==========================================================================
# 1. CONSTRUCCIONES
# ==========================================================================

def banda_equicolas(Xd: np.ndarray, nivel: float) -> Tuple[np.ndarray, np.ndarray]:
    """(S, ...) -> (li, ls) con la forma de `Xd[0]`: cuantiles zeta/2 y 1 - zeta/2."""
    li = np.quantile(Xd, (1 - nivel) / 2, axis=0)
    ls = np.quantile(Xd, 1 - (1 - nivel) / 2, axis=0)
    return li, ls


def banda_simultanea(Xd: np.ndarray, nivel: float, centro: Optional[np.ndarray] = None,
                     escala: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (S, n, G) -> (li, ls) (n, G). Banda simultanea c +- q s con q el cuantil `nivel`
    de D^(s) = max_g |X^(s) - c| / s. Por defecto c = mediana puntual y
    s = IQR / 1.349 puntual (robustos a las colas de atau <= 1).
    """
    X = np.asarray(Xd, dtype=float)
    c = np.median(X, axis=0) if centro is None else np.asarray(centro, float)
    if escala is None:
        q25, q75 = np.quantile(X, [0.25, 0.75], axis=0)
        s = (q75 - q25) / 1.349
    else:
        s = np.asarray(escala, float)
    s = np.where(s > 0, s, np.finfo(float).tiny)
    D = (np.abs(X - c[None]) / s[None]).max(axis=-1)                 # (S, n)
    q = np.quantile(D, nivel, axis=0)                                # (n,)
    return c - q[:, None] * s, c + q[:, None] * s


def _corridas(m: np.ndarray):
    """Corridas de True por fila de `m` (P, n): (fila, inicio, fin exclusivo), en orden."""
    d = np.diff(np.pad(m.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    p_ini, j_ini = np.nonzero(d == 1)
    p_fin, j_fin = np.nonzero(d == -1)
    assert np.array_equal(p_ini, p_fin)
    return p_ini, j_ini, j_fin


def _hdr_bloque(z: np.ndarray, nivel: float, n_grid: int, k_max: int, recorte):
    """HDR de cada columna de z (S, p). Retorna L, U (p, k_max) y fusiones (p,)."""
    S, p = z.shape
    lo, hi = np.quantile(z, list(recorte), axis=0)                   # (p,)
    rango = hi - lo
    degen = rango <= 1e-12 * np.maximum(1.0, np.abs(lo))
    delta = np.where(degen, 1.0, rango / n_grid)
    dentro = (z >= lo) & (z <= hi)
    j = np.clip(np.floor((z - lo) / delta).astype(np.int64), 0, n_grid - 1)
    col = np.broadcast_to(np.arange(p), (S, p))
    cuentas = np.bincount((col * n_grid + j)[dentro], minlength=p * n_grid).reshape(p, n_grid)
    n_in = cuentas.sum(axis=1)
    # Ancho de Scott sobre la muestra recortada, en unidades de celda. La densidad solo
    # ORDENA las celdas; la masa de la region es la empirica (Hyndman 1996), asi que el
    # suavizado no ensancha la region.
    zz = np.where(dentro, z, np.nan)
    sd = np.nanstd(zz, axis=0)
    h = sd * np.maximum(n_in, 1) ** (-0.2) / delta
    largo = int(2 ** np.ceil(np.log2(n_grid + 8.0 * float(np.max(h, initial=0.0)) + 1.0)))
    nu = np.fft.rfftfreq(largo)
    f = np.fft.irfft(np.fft.rfft(cuentas, largo, axis=1)
                     * np.exp(-2.0 * np.pi ** 2 * (h[:, None] * nu[None, :]) ** 2),
                     largo, axis=1)[:, :n_grid]
    orden = np.argsort(-f, axis=1, kind="stable")
    acum = np.cumsum(np.take_along_axis(cuentas, orden, axis=1), axis=1) / S
    alcanza = acum >= nivel
    k_umbral = np.where(alcanza.any(axis=1), np.argmax(alcanza, axis=1), n_grid - 1)
    umbral = np.take_along_axis(f, orden, axis=1)[np.arange(p), k_umbral]
    region = f >= umbral[:, None]

    L = np.full((p, k_max), np.nan)
    U = np.full((p, k_max), np.nan)
    fus = np.zeros(p, dtype=int)
    filas, j0, j1 = _corridas(region)
    n_tramos = np.bincount(filas, minlength=p)
    rango_fila = np.arange(len(filas)) - np.searchsorted(filas, filas)
    ok = n_tramos[filas] <= k_max
    L[filas[ok], rango_fila[ok]] = lo[filas[ok]] + j0[ok] * delta[filas[ok]]
    U[filas[ok], rango_fila[ok]] = lo[filas[ok]] + j1[ok] * delta[filas[ok]]
    for i in np.flatnonzero(n_tramos > k_max):
        a = list(lo[i] + j0[filas == i] * delta[i])
        b = list(lo[i] + j1[filas == i] * delta[i])
        while len(a) > k_max:
            g = int(np.argmin(np.array(a[1:]) - np.array(b[:-1])))
            b[g] = b.pop(g + 1)
            a.pop(g + 1)
            fus[i] += 1
        L[i], U[i] = a, b
    L[degen, 0], U[degen, 0] = lo[degen], hi[degen]
    L[degen, 1:], U[degen, 1:] = np.nan, np.nan
    return L, U, fus


def hdr_puntual(Xd: np.ndarray, nivel: float, n_grid: int = N_GRID_HDR, k_max: int = K_MAX_HDR,
                recorte: Optional[Sequence[float]] = None, devolver_fusiones: bool = False,
                bloque: int = 4096):
    """
    Region de maxima densidad de nivel `nivel` en cada tau_g (docs eq. hdr), sobre
    (S, n, G) -> L, U (n, G, k_max), NaN donde no hay tramo.

    Por punto: la muestra se lleva a un histograma de `n_grid` celdas entre sus
    cuantiles `recorte` ((0, 1) por defecto; `RECORTE_HDR` cuando
    `comportamientos.debe_recortar(hp_json)`, porque con atau <= 1 los extremos
    estirarian la grilla), y se suaviza con un nucleo gaussiano de ancho de Scott
    (por FFT, todas las (t, g) a la vez: no un `gaussian_kde` por punto). Las celdas
    se ordenan por densidad y se toman hasta acumular una fraccion `nivel` de las S
    extracciones (la masa es la empirica, como en Hyndman 1996: el suavizado ordena
    pero no ensancha); la region es la union de las corridas de celdas tomadas, cada
    una de borde a borde de celda. Con mas de `k_max` corridas se fusionan las dos mas
    cercanas, y se cuentan (`devolver_fusiones=True` retorna tambien un (n, G) con el
    numero de fusiones). Con una predictiva unimodal coincide con la equi-colas salvo
    el ancho de celda.
    """
    X = np.asarray(Xd, dtype=float)
    S, forma = X.shape[0], X.shape[1:]
    Z = X.reshape(S, -1)
    P = Z.shape[1]
    rec = (0.0, 1.0) if recorte is None else tuple(recorte)
    L, U, fus = np.empty((P, k_max)), np.empty((P, k_max)), np.empty(P, dtype=int)
    for p0 in range(0, P, bloque):
        sl = slice(p0, p0 + bloque)
        L[sl], U[sl], fus[sl] = _hdr_bloque(Z[:, sl], nivel, n_grid, k_max, rec)
    L, U = L.reshape(forma + (k_max,)), U.reshape(forma + (k_max,))
    return (L, U, fus.reshape(forma)) if devolver_fusiones else (L, U)


# ==========================================================================
# 2. METRICAS SOBRE UNA UNION DE TRAMOS
# ==========================================================================

def _tramos(y, L, U):
    y = np.asarray(y, float)
    L = np.asarray(L, float)
    U = np.asarray(U, float)
    if L.ndim == y.ndim:
        L, U = L[..., None], U[..., None]
    if not (L.shape == U.shape and L.shape[:-1] == y.shape):
        raise ValueError(f"Formas incompatibles: y {y.shape}, L {L.shape}, U {U.shape}.")
    if np.any(U < L):
        raise ValueError("Hay tramos con U < L.")
    return y, L, U


def ancho_tramos(L, U) -> np.ndarray:
    """sum_k (U_k - L_k) sobre el ULTIMO eje, el de los tramos: (..., K) -> (...).
    Una banda de un tramo se pasa con el eje agregado (`li[..., None]`)."""
    return np.nansum(np.asarray(U, float) - np.asarray(L, float), axis=-1)


def mpiw_tramos(L, U) -> float:
    """Promedio de `ancho_tramos` (con un tramo, `metrics_distribucional.mpiw`); (L, U)
    con el eje de los tramos al final."""
    return float(ancho_tramos(L, U).mean())


def winkler_tramos(y, L, U, nivel: float = 0.95) -> np.ndarray:
    """Winkler sobre una union de tramos, elemento a elemento (forma de `y`)."""
    if not 0.0 < nivel < 1.0:
        raise ValueError(f"nivel={nivel} debe estar en (0, 1).")
    y, L, U = _tramos(y, L, U)
    alpha = 1.0 - nivel
    yk = y[..., None]
    d = np.maximum(L - yk, 0.0) + np.maximum(yk - U, 0.0)
    return np.nansum(U - L, axis=-1) + (2.0 / alpha) * np.nanmin(d, axis=-1)


def cobertura_tramos(y, L, U) -> np.ndarray:
    """1{y en algun tramo}, sin agregar (forma de `y`)."""
    y, L, U = _tramos(y, L, U)
    yk = y[..., None]
    return ((yk >= L) & (yk <= U)).any(axis=-1)


def cobertura_simultanea_tramos(y, L, U) -> np.ndarray:
    """1 si todos los puntos del ultimo eje de `y` quedan en algun tramo."""
    return cobertura_tramos(y, L, U).all(axis=-1)


def resumen_intervalo_tramos(y, L, U, nivel: float = 0.95,
                             tau: Optional[np.ndarray] = None) -> dict:
    """`metrics_distribucional.resumen_intervalo` sobre una union de tramos."""
    W = winkler_tramos(y, L, U, nivel=nivel)
    _, L, U = _tramos(y, L, U)
    anchos = ancho_tramos(L, U)
    if tau is not None and W.ndim == 2:
        w = pesos_trapezoidales(tau)
        dominio = float(w.sum())
        w_val = float(np.mean(integrar(W, tau) / dominio))
        anc = float(np.mean(integrar(anchos, tau) / dominio))
    else:
        w_val, anc = float(W.mean()), float(anchos.mean())
    I = cobertura_tramos(y, L, U).astype(float)
    n, p = int(I.size), float(I.mean())
    return {"winkler": w_val, "picp": p,
            "ee_picp": float(np.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))),
            "ace": float(p - nivel), "mpiw": anc, "nivel": float(nivel), "n_efectivo": n,
            "primaria": "winkler"}
