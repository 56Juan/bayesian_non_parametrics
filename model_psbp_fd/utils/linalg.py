"""
linalg.py
=========
Algebra lineal numericamente estable.

Estas rutinas son la UNICA definicion de la factorizacion empleada en todo el
proyecto. Cualquier generacion de realizaciones gaussianas ---la innovacion
funcional de los generadores de simulacion, los pasos multivariados de un
muestreador--- debe construirse a partir de `safe_chol`: mantener dos
implementaciones con estrategias de regularizacion distintas produce
trayectorias distintas ante la misma semilla sin emitir error alguno.

Nota sobre el cambio de escala del jitter
-----------------------------------------
La version anterior de `safe_chol` empleaba un jitter ABSOLUTO (1e-8, x10 por
intento). Esa eleccion es fragil: si la matriz tiene entradas del orden de
1e-3 el jitter resulta enorme en terminos relativos, y si son del orden de
1e+3 resulta irrelevante. La version actual escala el jitter por la magnitud
media de la diagonal, de modo que la regularizacion es adimensional y su
efecto es el mismo con independencia de las unidades de la matriz.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.linalg import cholesky, LinAlgError

__all__ = ["sym", "safe_chol", "es_triangular_inferior"]


def sym(A: np.ndarray) -> np.ndarray:
    """Simetriza A para eliminar la asimetria residual de punto flotante."""
    A = np.asarray(A, dtype=float)
    return (A + A.T) / 2.0


def es_triangular_inferior(A: np.ndarray, tol: float = 1e-12) -> bool:
    """
    Indica si A es triangular inferior dentro de la tolerancia.

    Se expone porque el fallback espectral de `safe_chol` NO retorna un factor
    triangular: cualquier consumidor que resuelva sistemas por sustitucion
    hacia adelante en lugar de por producto matricial debe verificarlo antes.
    """
    A = np.asarray(A, dtype=float)
    return bool(np.abs(np.triu(A, k=1)).max() <= tol)


def safe_chol(
    A: np.ndarray,
    jitter: float = 1e-10,
    max_intentos: int = 12,
    fallback_espectral: bool = True,
) -> np.ndarray:
    """
    Factor L tal que L L^T ~= A, con regularizacion incremental relativa.

    Los nucleos de covarianza suaves ---en particular el exponencial
    cuadratico--- son notoriamente mal condicionados para grillas finas o
    longitudes de correlacion grandes, de modo que la factorizacion directa
    puede fallar aun cuando A sea teoricamente definida positiva. Se agrega a
    la diagonal un jitter proporcional a la magnitud media de la propia
    diagonal, escalado por potencias de diez hasta lograr la factorizacion.

    Parametros
    ----------
    jitter : float
        Jitter RELATIVO inicial. El termino efectivo en el intento k es
        `jitter * 10^k * mean(diag(A))`.
    max_intentos : int
        Numero de escalamientos sucesivos antes de recurrir al fallback.
    fallback_espectral : bool
        Si True y todos los intentos fallan, se emplea la descomposicion
        espectral truncando los valores propios negativos atribuibles a error
        numerico. El factor resultante satisface L L^T ~= A pero **no es
        triangular**; se emite un aviso porque el contrato cambia. Si es
        False, se propaga `LinAlgError`, comportamiento adecuado cuando el
        llamador prefiere detectar una especificacion invalida en lugar de
        continuar con una matriz regularizada.

    Retorna
    -------
    L : (n, n) factor tal que L L^T ~= A. Triangular inferior salvo en el
        fallback espectral.
    """
    A = sym(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A debe ser cuadrada; recibido {A.shape}.")

    n = A.shape[0]
    escala = float(np.mean(np.diag(A)))
    if not np.isfinite(escala) or escala <= 0.0:
        raise ValueError(
            "A tiene diagonal media no positiva o no finita; la matriz no "
            "puede ser una covarianza valida. Revise la especificacion."
        )

    for k in range(max_intentos):
        try:
            return cholesky(A + (jitter * (10.0 ** k) * escala) * np.eye(n))
        except LinAlgError:
            continue

    if not fallback_espectral:
        raise LinAlgError(
            f"safe_chol: la factorizacion no converge tras {max_intentos} "
            "escalamientos del jitter."
        )

    warnings.warn(
        "safe_chol: se recurrio al fallback espectral. El factor retornado "
        "satisface L L^T ~= A pero NO es triangular inferior; verifique con "
        "`es_triangular_inferior` si el consumidor asume triangularidad.",
        RuntimeWarning,
        stacklevel=2,
    )
    valores, vectores = np.linalg.eigh(A)
    valores = np.clip(valores, 0.0, None)
    return vectores * np.sqrt(valores)[None, :]
