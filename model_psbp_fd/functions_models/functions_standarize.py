import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple


class DataStandardizer:
    """
    Estandariza y desestandariza datos manteniendo registro de parámetros.
    
    Attributes:
        method: Método de estandarización ('zscore_column', 'minmax', etc.)
        mean: Media por columna (para zscore)
        std: Desv. estándar por columna (para zscore)
        ddof: Grados de libertad para std
        n_features: Número de features
    """
    
    SUPPORTED_METHODS = {"zscore_column", "minmax", "robust"}
    
    def __init__(self, method: str = "zscore_column", ddof: int = 0):
        """
        Args:
            method: Método de estandarización
            ddof: Grados de libertad para cálculo de std
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Método '{method}' no soportado. Usa: {self.SUPPORTED_METHODS}")
        
        self.method = method
        self.ddof = ddof
        self.is_fitted = False
        
        # Parámetros (se llenan con fit)
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.q25 = None
        self.q75 = None
        self.n_features = None
    
    def fit(self, X: np.ndarray) -> "DataStandardizer":
        """
        Calcula parámetros de estandarización.
        
        Args:
            X: Array de datos (n_samples, n_features)
        
        Returns:
            self (para encadenamiento)
        """
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, recibido: {X.ndim}D")
        
        self.n_features = X.shape[1]
        
        if self.method == "zscore_column":
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0, ddof=self.ddof)
            
            # Validar varianza nula
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
            
            iqr = self.q75 - self.q25
            if np.any(iqr == 0):
                cols_const = np.where(iqr == 0)[0].tolist()
                raise ValueError(f"Columnas con IQR nulo: {cols_const}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Estandariza los datos."""
        if not self.is_fitted:
            raise RuntimeError("Debe llamar .fit() primero")
        
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X tiene {X.shape[1]} features, se esperaban {self.n_features}"
            )
        
        if self.method == "zscore_column":
            return (X - self.mean) / self.std
        elif self.method == "minmax":
            return (X - self.min) / (self.max - self.min)
        elif self.method == "robust":
            iqr = self.q75 - self.q25
            return (X - self.median) / iqr
    
    def inverse_transform(self, X_std: np.ndarray) -> np.ndarray:
        """Desestandariza los datos (recupera escala original)."""
        if not self.is_fitted:
            raise RuntimeError("Debe llamar .fit() primero")
        
        if self.method == "zscore_column":
            return X_std * self.std + self.mean
        elif self.method == "minmax":
            return X_std * (self.max - self.min) + self.min
        elif self.method == "robust":
            iqr = self.q75 - self.q25
            return X_std * iqr + self.median
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit + transform en una llamada."""
        return self.fit(X).transform(X)
    
    def save(self, path: Path) -> None:
        """Guarda parámetros de estandarización."""
        if not self.is_fitted:
            raise RuntimeError("No hay nada que guardar. Llamar .fit() primero")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Guardar parámetros en NPZ
        save_dict = {
            "mean": self.mean if self.mean is not None else np.array([]),
            "std": self.std if self.std is not None else np.array([]),
            "min": self.min if self.min is not None else np.array([]),
            "max": self.max if self.max is not None else np.array([]),
            "q25": self.q25 if self.q25 is not None else np.array([]),
            "q75": self.q75 if self.q75 is not None else np.array([]),
            "median": self.median if hasattr(self, 'median') and self.median is not None else np.array([]),
        }
        
        np.savez_compressed(
            path / "standardizer_params.npz",
            **save_dict
        )
        
        # Guardar metadatos en JSON
        metadata = {
            "method": self.method,
            "ddof": self.ddof,
            "n_features": self.n_features,
        }
        
        with open(path / "standardizer_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Standardizer guardado en: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "DataStandardizer":
        """Carga parámetros de estandarización guardados."""
        path = Path(path)
        
        # Cargar metadatos
        with open(path / "standardizer_metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Crear instancia
        standardizer = cls(
            method=metadata["method"],
            ddof=metadata["ddof"]
        )
        standardizer.n_features = metadata["n_features"]
        
        # Cargar parámetros
        data = np.load(path / "standardizer_params.npz")
        
        if data["mean"].size > 0:
            standardizer.mean = data["mean"]
        if data["std"].size > 0:
            standardizer.std = data["std"]
        if data["min"].size > 0:
            standardizer.min = data["min"]
        if data["max"].size > 0:
            standardizer.max = data["max"]
        if data["q25"].size > 0:
            standardizer.q25 = data["q25"]
        if data["q75"].size > 0:
            standardizer.q75 = data["q75"]
        if data["median"].size > 0:
            standardizer.median = data["median"]
        
        standardizer.is_fitted = True
        print(f"✓ Standardizer cargado desde: {path}")
        
        return standardizer
    
    def summary(self) -> str:
        """Resumen de estadísticas de estandarización."""
        if not self.is_fitted:
            return "Standardizer no ajustado"
        
        lines = [
            f"Método: {self.method}",
            f"Features: {self.n_features}",
        ]
        
        if self.mean is not None:
            lines.append(f"  Media:     min={self.mean.min():.4f}, max={self.mean.max():.4f}")
            lines.append(f"  Std:       min={self.std.min():.4f}, max={self.std.max():.4f}")
        
        return "\n".join(lines)