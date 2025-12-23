"""
Módulo de simulación basado en Procesos Gaussianos para generar datos de regresión.

Este módulo permite generar covariables X mediante procesos gaussianos con diferentes
kernels y luego generar variables de respuesta Y mediante funciones de transformación
personalizables, manteniendo la estructura causal X -> Y.
"""

import numpy as np
from typing import Callable, Optional, Union, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ============================================================================
# KERNELS PARA PROCESOS GAUSSIANOS
# ============================================================================

class Kernel(ABC):
    """Clase base abstracta para kernels de procesos gaussianos."""
    
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de covarianza entre dos conjuntos de puntos.
        
        Parameters
        ----------
        X1 : np.ndarray
            Primer conjunto de puntos, shape (n1, d)
        X2 : np.ndarray
            Segundo conjunto de puntos, shape (n2, d)
            
        Returns
        -------
        np.ndarray
            Matriz de covarianza, shape (n1, n2)
        """
        pass


class RBFKernel(Kernel):
    """
    Kernel de función de base radial (RBF) o gaussiano.
    
    k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
    
    Parameters
    ----------
    length_scale : float
        Parámetro de escala de longitud (l)
    variance : float
        Varianza de la señal (σ²)
    """
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Calcular distancias euclidianas al cuadrado
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                 np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        
        return self.variance * np.exp(-0.5 * sqdist / self.length_scale**2)


class MaternKernel(Kernel):
    """
    Kernel de Matérn.
    
    Implementación para nu = 3/2:
    k(x, x') = σ² * (1 + √3*r/l) * exp(-√3*r/l)
    
    Parameters
    ----------
    length_scale : float
        Parámetro de escala de longitud
    variance : float
        Varianza de la señal
    nu : float
        Parámetro de suavidad (solo 1.5 y 2.5 implementados)
    """
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, nu: float = 1.5):
        self.length_scale = length_scale
        self.variance = variance
        self.nu = nu
        
        if nu not in [1.5, 2.5]:
            raise ValueError("Solo nu=1.5 y nu=2.5 están implementados")
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                 np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        r = np.sqrt(np.maximum(sqdist, 0))
        
        if self.nu == 1.5:
            scaled_r = np.sqrt(3) * r / self.length_scale
            return self.variance * (1 + scaled_r) * np.exp(-scaled_r)
        else:  # nu == 2.5
            scaled_r = np.sqrt(5) * r / self.length_scale
            return self.variance * (1 + scaled_r + scaled_r**2 / 3) * np.exp(-scaled_r)


class PeriodicKernel(Kernel):
    """
    Kernel periódico.
    
    k(x, x') = σ² * exp(-2 * sin²(π|x-x'|/p) / l²)
    
    Parameters
    ----------
    period : float
        Período de la función
    length_scale : float
        Parámetro de escala de longitud
    variance : float
        Varianza de la señal
    """
    
    def __init__(self, period: float = 1.0, length_scale: float = 1.0, variance: float = 1.0):
        self.period = period
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        dists = np.abs(X1[:, :, None] - X2.T[None, :, :])
        dists = np.sum(dists, axis=1)
        
        sin_component = np.sin(np.pi * dists / self.period)
        return self.variance * np.exp(-2 * sin_component**2 / self.length_scale**2)


class LinearKernel(Kernel):
    """
    Kernel lineal.
    
    k(x, x') = σ² * (x - c)ᵀ(x' - c)
    
    Parameters
    ----------
    variance : float
        Varianza de la señal
    offset : float
        Punto de desplazamiento
    """
    
    def __init__(self, variance: float = 1.0, offset: float = 0.0):
        self.variance = variance
        self.offset = offset
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1) - self.offset
        X2 = np.atleast_2d(X2) - self.offset
        
        return self.variance * np.dot(X1, X2.T)


# ============================================================================
# GENERADOR DE PROCESOS GAUSSIANOS
# ============================================================================

class GaussianProcess:
    """
    Generador de muestras de un proceso gaussiano.
    
    Parameters
    ----------
    kernel : Kernel
        Función de kernel para el proceso gaussiano
    mean_function : Callable, optional
        Función de media, por defecto 0
    noise_variance : float
        Varianza del ruido de observación
    random_state : int, optional
        Semilla para reproducibilidad
    """
    
    def __init__(
        self,
        kernel: Kernel,
        mean_function: Optional[Callable] = None,
        noise_variance: float = 1e-8,
        random_state: Optional[int] = None
    ):
        self.kernel = kernel
        self.mean_function = mean_function if mean_function else lambda x: np.zeros(len(x))
        self.noise_variance = noise_variance
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def sample(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Genera muestras del proceso gaussiano en los puntos X.
        
        Parameters
        ----------
        X : np.ndarray
            Puntos donde evaluar el GP, shape (n_points, n_dims)
        n_samples : int
            Número de muestras a generar
            
        Returns
        -------
        np.ndarray
            Muestras del GP, shape (n_points, n_samples)
        """
        X = np.atleast_2d(X)
        n_points = X.shape[0]
        
        # Calcular matriz de covarianza
        K = self.kernel(X, X)
        K += self.noise_variance * np.eye(n_points)  # Estabilidad numérica
        
        # Calcular media
        mean = self.mean_function(X).reshape(-1, 1)
        
        # Generar muestras
        L = np.linalg.cholesky(K)
        samples = mean + L @ self.rng.randn(n_points, n_samples)
        
        return samples


# ============================================================================
# SIMULADOR DE REGRESIÓN
# ============================================================================

@dataclass
class SimulationConfig:
    """
    Configuración para la simulación de datos de regresión.
    
    Parameters
    ----------
    n_samples : int
        Número de observaciones
    n_features : int
        Número de covariables
    x_range : Tuple[float, float]
        Rango de valores para generar X
    noise_std : float
        Desviación estándar del ruido en Y
    random_state : int, optional
        Semilla para reproducibilidad
    """
    n_samples: int
    n_features: int
    x_range: Tuple[float, float] = (0.0, 10.0)
    noise_std: float = 0.1
    random_state: Optional[int] = None


class RegressionSimulator:
    """
    Simulador de datos de regresión usando procesos gaussianos.
    
    Este simulador genera covariables X mediante procesos gaussianos y luego
    aplica una función de transformación para generar Y = f(X) + ε, manteniendo
    la estructura causal X -> Y.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuración de la simulación
    kernel : Kernel
        Kernel para generar las covariables X
    transformation : Callable
        Función f que transforma X en Y
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        kernel: Kernel,
        transformation: Callable[[np.ndarray], np.ndarray]
    ):
        self.config = config
        self.kernel = kernel
        self.transformation = transformation
        self.rng = np.random.RandomState(config.random_state)
        
        # Inicializar proceso gaussiano
        self.gp = GaussianProcess(
            kernel=kernel,
            random_state=config.random_state
        )
    
    def generate_covariates(self) -> np.ndarray:
        """
        Genera las covariables X mediante procesos gaussianos.
        
        Returns
        -------
        np.ndarray
            Matriz de covariables, shape (n_samples, n_features)
        """
        # Generar puntos de entrada en el rango especificado
        x_min, x_max = self.config.x_range
        input_points = np.linspace(x_min, x_max, self.config.n_samples).reshape(-1, 1)
        
        # Generar cada covariable como una muestra del GP
        X = np.zeros((self.config.n_samples, self.config.n_features))
        
        for i in range(self.config.n_features):
            # Cada característica es una realización independiente del GP
            sample = self.gp.sample(input_points, n_samples=1)
            X[:, i] = sample.flatten()
        
        return X
    
    def generate_response(self, X: np.ndarray) -> np.ndarray:
        """
        Genera la variable de respuesta Y = f(X) + ε.
        
        Parameters
        ----------
        X : np.ndarray
            Matriz de covariables, shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Vector de respuestas, shape (n_samples,)
        """
        # Aplicar la transformación f(X)
        Y_clean = self.transformation(X)
        
        # Agregar ruido gaussiano
        noise = self.rng.normal(0, self.config.noise_std, size=Y_clean.shape)
        Y = Y_clean + noise
        
        return Y
    
    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta la simulación completa.
        
        Returns
        -------
        X : np.ndarray
            Matriz de covariables, shape (n_samples, n_features)
        Y : np.ndarray
            Vector de respuestas, shape (n_samples,)
        """
        X = self.generate_covariates()
        Y = self.generate_response(X)
        
        return X, Y


# ============================================================================
# FUNCIONES DE TRANSFORMACIÓN PREDEFINIDAS
# ============================================================================

class TransformationFunctions:
    """Colección de funciones de transformación comunes."""
    
    @staticmethod
    def linear(coefficients: np.ndarray, intercept: float = 0.0) -> Callable:
        """
        Transformación lineal: y = β₀ + β₁x₁ + ... + βₚxₚ
        
        Parameters
        ----------
        coefficients : np.ndarray
            Coeficientes de regresión
        intercept : float
            Término de intercepto
        """
        def transform(X: np.ndarray) -> np.ndarray:
            return intercept + X @ coefficients
        return transform
    
    @staticmethod
    def polynomial(degree: int = 2) -> Callable:
        """
        Transformación polinomial: y = x₁² + x₂² + ...
        
        Parameters
        ----------
        degree : int
            Grado del polinomio
        """
        def transform(X: np.ndarray) -> np.ndarray:
            return np.sum(X**degree, axis=1)
        return transform
    
    @staticmethod
    def interaction() -> Callable:
        """
        Transformación con interacciones: y = x₁ * x₂ + x₁ * x₃ + ...
        """
        def transform(X: np.ndarray) -> np.ndarray:
            n_features = X.shape[1]
            result = np.zeros(X.shape[0])
            for i in range(n_features - 1):
                result += X[:, i] * X[:, i + 1]
            return result
        return transform
    
    @staticmethod
    def nonlinear_combination() -> Callable:
        """
        Combinación no lineal: y = sin(x₁) + log(|x₂| + 1) + exp(x₃/10)
        """
        def transform(X: np.ndarray) -> np.ndarray:
            result = np.zeros(X.shape[0])
            n_features = X.shape[1]
            
            if n_features >= 1:
                result += np.sin(X[:, 0])
            if n_features >= 2:
                result += np.log(np.abs(X[:, 1]) + 1)
            if n_features >= 3:
                result += np.exp(X[:, 2] / 10)
            
            return result
        return transform
    
    @staticmethod
    def custom(func: Callable) -> Callable:
        """
        Permite especificar una función personalizada.
        
        Parameters
        ----------
        func : Callable
            Función que toma X (n_samples, n_features) y retorna y (n_samples,)
        """
        return func


