"""
pooling.py
==========
Agrupacion de cadenas MCMC independientes.

Cuando se ejecutan varias cadenas sobre el mismo modelo y datos, la
distribucion predictiva agrupada es la mezcla de igual peso de las predictivas
por cadena. Emplear una sola cadena descarta la fraccion correspondiente del
posterior sin ganancia alguna, y promediar solo las medias subestima la
incertidumbre al ignorar la dispersion entre cadenas.
"""

from __future__ import annotations

import numpy as np

__all__ = ["agrupar_momentos", "agrupar_muestras"]


def agrupar_momentos(medias: np.ndarray, desviaciones: np.ndarray):
    """
    Media y desviacion de la mezcla de igual peso de varias cadenas.

    Para una mezcla de C componentes con igual peso, la ley de varianza total
    descompone la varianza de la mezcla en la varianza intra-cadena promedio
    mas la varianza de las medias entre cadenas:

        Var_pool = E_c[sigma_c^2] + Var_c[mu_c].

    Omitir el segundo termino ---es decir, promediar unicamente las medias y
    las desviaciones--- subestima la incertidumbre, y esa subestimacion es
    tanto mayor cuanto peor sea la mezcla de las cadenas, justo cuando mas
    importa detectarla.

    medias, desviaciones : (n, C) por origen y cadena.
    Retorna (mu_pool, sd_pool), ambos (n,).
    """
    M = np.atleast_2d(np.asarray(medias, dtype=float))
    S = np.atleast_2d(np.asarray(desviaciones, dtype=float))
    if M.shape != S.shape:
        raise ValueError(f"Formas incompatibles: medias {M.shape}, desv. {S.shape}.")
    if np.any(S < 0):
        raise ValueError("Las desviaciones no pueden ser negativas.")

    mu = M.mean(axis=1)
    var = (S ** 2).mean(axis=1) + ((M - mu[:, None]) ** 2).mean(axis=1)
    return mu, np.sqrt(var)


def agrupar_muestras(lista_muestras) -> np.ndarray:
    """
    Concatena las extracciones de varias cadenas en una sola muestra.

    Es la representacion exacta de la mezcla cuando las cadenas tienen igual
    longitud tras el descarte inicial; con longitudes distintas la mezcla queda
    ponderada por el numero de extracciones, por lo que se verifica la
    homogeneidad y se advierte en caso contrario.

    lista_muestras : lista de arreglos (S_c, ...) con dimensiones no iniciales
                     identicas entre cadenas.
    """
    arrs = [np.asarray(a, dtype=float) for a in lista_muestras]
    if not arrs:
        raise ValueError("La lista de muestras esta vacia.")
    formas = {a.shape[1:] for a in arrs}
    if len(formas) > 1:
        raise ValueError(f"Dimensiones no iniciales heterogeneas: {formas}.")
    longitudes = {a.shape[0] for a in arrs}
    if len(longitudes) > 1:
        import warnings
        warnings.warn(
            f"Cadenas de longitud distinta {sorted(longitudes)}: la mezcla "
            "resultante queda ponderada por el numero de extracciones.",
            RuntimeWarning,
        )
    return np.concatenate(arrs, axis=0)
