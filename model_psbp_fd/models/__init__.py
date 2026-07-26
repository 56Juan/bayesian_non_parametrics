"""
models
======
Versiones del modelo PSBP-FD. Ver `CHANGELOG.md` para el detalle de cada una.

    psbp_fd_v1  Version preliminar. Reproduce Chung & Dunson (2009) con
                muestreador y predictor en Python. Incluye una extension
                compilada especifica de plataforma.
    psbp_fd_v2  Version revisada del muestreador y del predictor. La
                estandarizacion pasa a ser externa. Su predictor reporta la
                desviacion posterior de la media condicional, NO la
                predictiva; ver v3.
    psbp_fd_v3  Inferencia predictiva funcional. No contiene muestreador: el
                ajuste ocurre en MATLAB (`psbp_train.m`) y esta capa consume
                las trazas para construir la predictiva completa y
                transportarla del espacio de los scores al de las curvas.
                Es la version en uso.

Sobre la importacion de las versiones antiguas
----------------------------------------------
`psbp_fd_v1` depende de una extension compilada (`.pyd`) especifica de
plataforma y de arquitectura. Importarla de forma incondicional hace que
`import model_psbp_fd.models` falle por completo en cualquier entorno distinto
de aquel en que se compilo, arrastrando consigo a `psbp_fd_v3`, que no tiene
esa dependencia y es la version efectivamente en uso.

Las versiones heredadas se importan por tanto de forma tolerante, pero NO
silenciosa: el motivo de la indisponibilidad queda registrado en
`VERSIONES_NO_DISPONIBLES` y puede consultarse con `estado_versiones()`. La
distincion importa: un `except ImportError: pass` esconderia un error real de
codigo bajo la misma alfombra que una incompatibilidad de plataforma, mientras
que aqui ambos casos quedan visibles y distinguibles.

`psbp_fd_v3` se importa de forma estricta: si falla, debe fallar ruidosamente,
porque es la version que el estudio utiliza.
"""

from typing import Dict

# ── Version en uso: importacion estricta ─────────────────────────────────
from .pspb_fd_v3 import PSBP_FD_v3

# ── Versiones heredadas: importacion tolerante y trazable ────────────────
VERSIONES_NO_DISPONIBLES: Dict[str, str] = {}

try:
    from .psbp_fd_v1 import PSBP_FD_v1
except Exception as exc:  # extension compilada ausente o incompatible
    PSBP_FD_v1 = None
    VERSIONES_NO_DISPONIBLES["psbp_fd_v1"] = f"{type(exc).__name__}: {exc}"

try:
    from .psbp_fd_v2 import PSBP_FD_v2
except Exception as exc:
    PSBP_FD_v2 = None
    VERSIONES_NO_DISPONIBLES["psbp_fd_v2"] = f"{type(exc).__name__}: {exc}"


def estado_versiones() -> Dict[str, str]:
    """
    Estado de importacion de cada version del modelo.

    Retorna {version: "disponible"} o {version: motivo del fallo}, de modo que
    una version ausente pueda diagnosticarse sin releer el arbol de imports.
    """
    estado = {"psbp_fd_v3": "disponible"}
    for nombre, clase in (("psbp_fd_v1", PSBP_FD_v1), ("psbp_fd_v2", PSBP_FD_v2)):
        estado[nombre] = ("disponible" if clase is not None
                          else VERSIONES_NO_DISPONIBLES.get(nombre, "no disponible"))
    return estado


__all__ = [
    "PSBP_FD_v3",
    "PSBP_FD_v2",
    "PSBP_FD_v1",
    "VERSIONES_NO_DISPONIBLES",
    "estado_versiones",
]
