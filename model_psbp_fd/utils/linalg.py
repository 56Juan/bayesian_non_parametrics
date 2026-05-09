"""
linalg.py
---------
Utilidades de álgebra lineal numéricamente estables.
"""
import numpy as np
from numpy.linalg import cholesky, LinAlgError


def sym(A: np.ndarray) -> np.ndarray:
    """Simetriza A para eliminar errores de punto flotante."""
    return (A + A.T) / 2


def safe_chol(A: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Cholesky con jitter incremental si A no es positiva definida."""
    A = sym(A)
    for _ in range(5):
        try:
            return cholesky(A)
        except LinAlgError:
            A += jitter * np.eye(A.shape[0])
            jitter *= 10
    raise LinAlgError("safe_chol: no converge con jitter máximo.")
