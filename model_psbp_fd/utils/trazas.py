r"""
trazas.py — Normalizacion de forma de las trazas `.mat` que escribe MATLAB
==========================================================================

MATLAB **descarta las dimensiones singleton finales**: un arreglo declarado
como `zeros(nsim, N, p)` con `p = 1` se guarda en el `.mat` como `nsim x N`, no
como `nsim x N x 1`. `scipy.io.loadmat` lo lee tal cual, de modo que del lado
de Python la traza llega con dos ejes en vez de tres.

Eso ocurre exactamente cuando el diseno tiene **una sola covariable**, es decir
cuando `p = q * M = 1`: el punto `M = 1` del barrido de la Etapa D. Todo el
codigo aguas abajo asume `(nsim, N, p)` --`PSBPPredictor` lee `shape[2]` para
fijar `n_features_`, `pip_por_componente` promedia sobre el eje 0 esperando
`(N, p)`-- y falla con `IndexError: tuple index out of range` o devuelve una
matriz de una dimension menos. No es un problema del muestreador ni de los
datos: la traza es correcta, le falta el eje.

`normalizar_trazas_mat` restaura ese eje. La reconstruccion es inequivoca: el
numero de atomos `N` lo fija `beta0hout`, que es `(nsim, N)` por definicion y
nunca pierde ejes, asi que un arreglo 2D en las claves de tres ejes solo puede
ser el caso `p = 1`. Se aplica al leer, antes de cualquier consumo, para que el
resto del proyecto siga teniendo una sola convencion de formas.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["normalizar_trazas_mat", "asegurar_3d"]

# Claves cuya forma canonica es (nsim, *, p) y que por tanto pierden el ultimo
# eje cuando p = 1.
_CLAVES_3D = ("betajhout", "psijhout", "Gammajhout", "gammajhout")

# Claves cuya forma canonica es (nsim, p): con p = 1 MATLAB las guarda como
# (nsim, 1), que ya es 2D, de modo que no hay nada que restaurar.
# Se listan solo para dejar constancia de por que NO estan en _CLAVES_3D.
_CLAVES_2D = ("osumout", "pijout", "wjout")


def asegurar_3d(a: np.ndarray) -> np.ndarray:
    """Devuelve `a` con tres ejes, agregando el ultimo si viene colapsado."""
    a = np.asarray(a)
    if a.ndim == 2:
        return a[:, :, None]
    return a


def normalizar_trazas_mat(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Restaura el eje de covariables que MATLAB descarta cuando `p = 1`.

    Parametros
    ----------
    traces : dict
        Trazas tal como salen de `scipy.io.loadmat`.

    Retorna
    -------
    dict
        Copia superficial con las claves de `_CLAVES_3D` en forma `(nsim, *, p)`.
        Las demas claves se pasan sin tocar.
    """
    salida = dict(traces)
    for clave in _CLAVES_3D:
        if clave in salida and salida[clave] is not None:
            salida[clave] = asegurar_3d(salida[clave])
    return salida
