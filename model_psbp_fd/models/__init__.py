"""
models
======
Version en uso del modelo: `psbp_fd_v3` (inferencia predictiva funcional).

No contiene muestreador: el ajuste ocurre en MATLAB (`psbp_train.m`) y esta capa
consume las trazas para construir la predictiva completa y transportarla del
espacio de los scores al de las curvas.

`psbp_fd_v1` y `psbp_fd_v2` siguen en disco como historia pero ya no se importan
desde aqui (v1 depende de una extension compilada especifica de plataforma). Si
hiciera falta alguna, importarla directamente desde su modulo.
"""

from .pspb_fd_v3 import PSBP_FD_v3

__all__ = ["PSBP_FD_v3"]
