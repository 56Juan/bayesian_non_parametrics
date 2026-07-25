"""
functions_standarize.py
=======================
Estandarizacion de scores con persistencia de parametros.

Registro del bloque de ajuste
-----------------------------
Bajo el esquema de retencion temporal, la correccion del estudio depende de que
este objeto se ajuste EXCLUSIVAMENTE con el bloque de entrenamiento
{1, ..., T0} y se aplique despues, via `transform`, a la serie completa. En la
version anterior esa garantia era puramente nominal: la clase no guardaba
ningun rastro de con que datos habia sido ajustada, de modo que un ajuste
accidental sobre la serie completa producia resultados sin error alguno y sin
posibilidad de detectarlo despues. Con un solo escenario eso era vigilable a
ojo; con R replicas por escenario deja de serlo.

La clase registra ahora `n_ajuste` (numero de filas del bloque de ajuste) y
`etiqueta_ajuste` (descripcion libre del bloque), ambos persistidos en los
metadatos. Eso convierte la disciplina de retencion en una propiedad
VERIFICABLE: el notebook o `verificar_contrato` pueden comprobar
`estandarizador.n_ajuste == T0` en lugar de confiar en que la celda correcta se
ejecuto en el orden correcto.

Silencio por defecto
--------------------
`save` y `load` ya no imprimen. Con R replicas por escenario, la traza de
progreso generaba miles de lineas sin valor diagnostico. Quien la quiera puede
activarla con `verbose=True`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["DataStandardizer"]


class DataStandardizer:
    """
    Estandariza y desestandariza datos manteniendo registro de parametros.

    Atributos
    ---------
    method          : metodo de estandarizacion ('zscore_column', 'minmax',
                      'robust').
    mean, std       : momentos por columna (zscore).
    min, max        : extremos por columna (minmax).
    q25, q75, median: cuantiles por columna (robust).
    ddof            : grados de libertad de la desviacion estandar.
    n_features      : numero de columnas del bloque de ajuste.
    n_ajuste        : numero de FILAS del bloque de ajuste. Bajo retencion
                      temporal debe coincidir con T0.
    etiqueta_ajuste : descripcion del bloque empleado en `fit` (por ejemplo
                      "train[1:117]"). Se persiste y permite auditar la
                      procedencia del ajuste.
    verbose         : si True, `save` y `load` informan por consola.
    """

    SUPPORTED_METHODS = {"zscore_column", "minmax", "robust"}

    def __init__(self, method: str = "zscore_column", ddof: int = 0,
                 verbose: bool = False):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Metodo '{method}' no soportado. Usa: {self.SUPPORTED_METHODS}"
            )

        self.method = method
        self.ddof = ddof
        self.verbose = bool(verbose)
        self.is_fitted = False

        # Parametros (se llenan con fit)
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.q25 = None
        self.q75 = None
        self.median = None
        self.n_features = None

        # Registro del bloque de ajuste
        self.n_ajuste = None
        self.etiqueta_ajuste = None

    # ------------------------------------------------------------------
    # AJUSTE
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray,
            etiqueta: Optional[str] = None) -> "DataStandardizer":
        """
        Calcula los parametros de estandarizacion sobre el bloque de ajuste.

        Parametros
        ----------
        X : (n_ajuste, n_features)
            Bajo retencion temporal debe contener UNICAMENTE el bloque de
            entrenamiento. La clase no puede comprobarlo por si sola, pero
            registra `n_ajuste` para que el llamador si pueda.
        etiqueta : str, opcional
            Descripcion del bloque, p. ej. "train[1:117]" o "escenario1_rep03".
            Se persiste junto a los parametros.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, recibido: {X.ndim}D")

        self.n_features = X.shape[1]
        self.n_ajuste = X.shape[0]
        self.etiqueta_ajuste = etiqueta

        if self.method == "zscore_column":
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0, ddof=self.ddof)
            if np.any(self.std == 0):
                cols_zero = np.where(self.std == 0)[0].tolist()
                raise ValueError(f"Columnas con varianza nula: {cols_zero}")

        elif self.method == "minmax":
            self.min = X.min(axis=0)
            self.max = X.max(axis=0)
            if np.any(self.max == self.min):
                cols_const = np.where(self.max == self.min)[0].tolist()
                raise ValueError(f"Columnas constantes: {cols_const}")

        elif self.method == "robust":
            self.q25 = np.percentile(X, 25, axis=0)
            self.q75 = np.percentile(X, 75, axis=0)
            self.median = np.median(X, axis=0)
            if np.any((self.q75 - self.q25) == 0):
                cols_const = np.where((self.q75 - self.q25) == 0)[0].tolist()
                raise ValueError(f"Columnas con IQR nulo: {cols_const}")

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # APLICACION
    # ------------------------------------------------------------------
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Debe llamar .fit() primero")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, recibido: {X.ndim}D")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X tiene {X.shape[1]} features, se esperaban {self.n_features}"
            )
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Estandariza los datos con los parametros del bloque de ajuste.

        Admite bloques de entrenamiento o de prueba indistintamente: los
        momentos empleados son siempre los del ajuste, por lo que aplicar este
        metodo al bloque de prueba no introduce fuga de informacion.
        """
        X = self._check_X(X)
        if self.method == "zscore_column":
            return (X - self.mean) / self.std
        if self.method == "minmax":
            return (X - self.min) / (self.max - self.min)
        if self.method == "robust":
            return (X - self.median) / (self.q75 - self.q25)
        raise RuntimeError(f"Metodo no implementado: {self.method}")

    def inverse_transform(self, X_std: np.ndarray) -> np.ndarray:
        """Desestandariza los datos (recupera la escala original)."""
        X_std = self._check_X(X_std)
        if self.method == "zscore_column":
            return X_std * self.std + self.mean
        if self.method == "minmax":
            return X_std * (self.max - self.min) + self.min
        if self.method == "robust":
            return X_std * (self.q75 - self.q25) + self.median
        raise RuntimeError(f"Metodo no implementado: {self.method}")

    def fit_transform(self, X: np.ndarray,
                      etiqueta: Optional[str] = None) -> np.ndarray:
        """Fit + transform en una llamada, sobre el bloque de ajuste."""
        return self.fit(X, etiqueta=etiqueta).transform(X)

    # ------------------------------------------------------------------
    # VERIFICACION DE LA RETENCION TEMPORAL
    # ------------------------------------------------------------------
    def verificar_ajuste(self, n_esperado: int, estricto: bool = True) -> dict:
        """
        Comprueba que el ajuste se realizo sobre el numero de filas esperado.

        Se invoca con T0 para confirmar que el estandarizador no vio el bloque
        de prueba. Es la contraparte verificable de una disciplina que antes
        dependia por completo del orden de ejecucion de las celdas.
        """
        if not self.is_fitted:
            raise RuntimeError("Debe llamar .fit() primero")
        ok = (self.n_ajuste == int(n_esperado))
        informe = {
            "n_ajuste": int(self.n_ajuste),
            "n_esperado": int(n_esperado),
            "etiqueta_ajuste": self.etiqueta_ajuste,
            "ajuste_ok": bool(ok),
        }
        if not ok and estricto:
            raise ValueError(
                f"El estandarizador se ajusto con {self.n_ajuste} filas y se "
                f"esperaban {n_esperado}. Si {self.n_ajuste} corresponde a la "
                "serie completa, el bloque de prueba contamino los momentos de "
                "estandarizacion y los resultados fuera de muestra no son "
                "validos."
            )
        return informe

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Guarda los parametros de estandarizacion."""
        if not self.is_fitted:
            raise RuntimeError("No hay nada que guardar. Llamar .fit() primero")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        vacio = np.array([])
        save_dict = {
            "mean": self.mean if self.mean is not None else vacio,
            "std": self.std if self.std is not None else vacio,
            "min": self.min if self.min is not None else vacio,
            "max": self.max if self.max is not None else vacio,
            "q25": self.q25 if self.q25 is not None else vacio,
            "q75": self.q75 if self.q75 is not None else vacio,
            "median": self.median if self.median is not None else vacio,
        }
        np.savez_compressed(path / "standardizer_params.npz", **save_dict)

        metadata = {
            "method": self.method,
            "ddof": self.ddof,
            "n_features": int(self.n_features),
            "n_ajuste": int(self.n_ajuste),
            "etiqueta_ajuste": self.etiqueta_ajuste,
        }
        with open(path / "standardizer_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"Standardizer guardado en: {path}")

    @classmethod
    def load(cls, path: Path, verbose: bool = False) -> "DataStandardizer":
        """Carga parametros de estandarizacion previamente guardados."""
        path = Path(path)

        with open(path / "standardizer_metadata.json", "r") as f:
            metadata = json.load(f)

        std = cls(method=metadata["method"], ddof=metadata["ddof"],
                  verbose=verbose)
        std.n_features = metadata["n_features"]
        # Compatibilidad con artefactos generados antes del registro del bloque
        # de ajuste: su ausencia se declara como desconocida y no como cero,
        # para que `verificar_ajuste` no de un falso negativo silencioso.
        std.n_ajuste = metadata.get("n_ajuste")
        std.etiqueta_ajuste = metadata.get("etiqueta_ajuste")

        data = np.load(path / "standardizer_params.npz")
        for nombre in ("mean", "std", "min", "max", "q25", "q75", "median"):
            if data[nombre].size > 0:
                setattr(std, nombre, data[nombre])

        std.is_fitted = True
        if verbose:
            print(f"Standardizer cargado desde: {path}")
        return std

    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Resumen de las estadisticas de estandarizacion."""
        if not self.is_fitted:
            return "Standardizer no ajustado"

        lines = [
            f"Metodo: {self.method}",
            f"Features: {self.n_features}",
            f"Filas de ajuste: {self.n_ajuste}"
            + (f"  ({self.etiqueta_ajuste})" if self.etiqueta_ajuste else ""),
        ]
        if self.mean is not None:
            lines.append(
                f"  Media: min={self.mean.min():.4f}, max={self.mean.max():.4f}")
            lines.append(
                f"  Std:   min={self.std.min():.4f}, max={self.std.max():.4f}")
        return "\n".join(lines)
