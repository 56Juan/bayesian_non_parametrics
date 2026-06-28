"""
simulation_pipeline_nonlinear.py
================================
Simulador de series de tiempo funcionales FAR(p) NO LINEAL con tendencia
determinística. Extiende `simulation_pipeline_linear.py` en dos direcciones:

    (i)  orden p arbitrario (p rezagos en la recurrencia dinámica), y
    (ii) cuatro familias de no-linealidad intercambiables.

Recurrencia general
-------------------
    X_t(s) = μ_t(s) + Y_t(s)
    Y_t(s) = F( Y_{t-1}, ..., Y_{t-p} )(s) + ε_t(s)

donde F : (ℝ^T)^p → ℝ^T es un MAPA NO LINEAL especificado por uno de los
cuatro módulos (clases `*FAR`) definidos abajo. El término lineal de cada
módulo se construye con los mismos operadores integrales Ψ_k (T×T) que el
caso lineal, uno por rezago, evaluados por cuadratura rectangular.

Estacionariedad
---------------
Para el caso lineal la condición es ρ(C) < 1, con C la matriz companion de
bloques. Bajo no-linealidad esa condición deja de ser necesaria/suficiente
en general; cada módulo expone `stability_report()` con la cota apropiada:

    - Tramos lineales (umbral, mixtura): se reporta el RADIO ESPECTRAL de la
      companion por régimen/componente y se toma el máximo (condición
      suficiente de tipo SETAR).
    - No-linealidad Lipschitz (puntual): se reporta la CONSTANTE DE
      CONTRACCIÓN Σ_k L_g·||Ψ_k||_2 (condición suficiente de punto fijo de
      Banach sobre el estado apilado con norma del máximo).
    - Bilineal: NO es globalmente Lipschitz salvo saturación; sin saturación
      sólo se reporta el radio espectral de la parte lineal y se advierte que
      no hay garantía global.

Nota de diseño
--------------
Se reutilizan `FunctionalDomain`, `gaussian_integral_kernel`,
`build_integral_matrix` y la lógica de tendencia/ruido del módulo lineal para
mantener un contrato idéntico en el pipeline (CSV/JSON aguas abajo). Por ello
este módulo IMPORTA de `simulation_pipeline_linear`.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm as _stdnorm
from scipy.stats import normaltest

# Reutilización del módulo lineal (mismo contrato de dominio/kernel/tendencia).
# Import relativo dentro del paquete; absoluto como respaldo al ejecutar el
# archivo directamente como script (bloque __main__).
try:
    from .simulation_pipeline_linear import (
        FunctionalDomain,
        build_integral_matrix,
        gaussian_integral_kernel,
        FARSimulator,
    )
except ImportError:
    from simulation_pipeline_linear import (
        FunctionalDomain,
        build_integral_matrix,
        gaussian_integral_kernel,
        FARSimulator,
    )

__all__ = [
    "build_lag_operators",
    "companion_spectral_radius",
    "NonlinearFARMap",
    "ThresholdFAR",
    "PointwiseNonlinearFAR",
    "BilinearFAR",
    "MixtureStateFAR",
    "NonlinearFARSimulator",
]


# ============================================================
# UTILIDADES DE OPERADORES Y ESTABILIDAD
# ============================================================

def build_lag_operators(
    domain: FunctionalDomain,
    p: int,
    kernel_fn: Callable = gaussian_integral_kernel,
    decays: Optional[Sequence[float]] = None,
    base_decay: float = 0.6,
    decay_ratio: float = 0.5,
    **kernel_kwargs,
) -> list[np.ndarray]:
    """
    Construye una lista de p operadores integrales Ψ_1, ..., Ψ_p (uno por rezago).

    Si `decays` no se entrega, se usa un decaimiento GEOMÉTRICO por rezago:
        decay_k = base_decay · decay_ratio^{k-1},  k = 1, ..., p
    de modo que rezagos más antiguos pesan menos (parametrización conservadora
    que tiende a favorecer la estacionariedad). El `decay` de cada operador se
    pasa directamente al kernel gaussiano.

    Parámetros
    ----------
    domain : FunctionalDomain
    p : int
        Número de rezagos (orden del FAR(p)). Debe ser >= 1.
    kernel_fn : Callable
        Kernel del operador integral. Por defecto el gaussiano del módulo lineal.
    decays : secuencia de float, opcional
        Decaimientos explícitos por rezago. Si se da, su longitud debe ser p
        y anula `base_decay`/`decay_ratio`.
    base_decay, decay_ratio : float
        Parámetros del decaimiento geométrico por defecto.
    **kernel_kwargs
        Argumentos extra del kernel (p.ej. bandwidth). NO incluir `decay`.

    Retorna
    -------
    list[np.ndarray] de longitud p, cada uno (n_points × n_points).
    """
    if p < 1:
        raise ValueError("p debe ser >= 1.")
    if "decay" in kernel_kwargs:
        raise ValueError("No pase 'decay' en kernel_kwargs; use `decays` o `base_decay`.")
    if decays is None:
        decays = [base_decay * (decay_ratio ** k) for k in range(p)]
    else:
        decays = list(decays)
        if len(decays) != p:
            raise ValueError(f"len(decays)={len(decays)} != p={p}.")
    return [
        build_integral_matrix(domain, kernel_fn, decay=float(d), **kernel_kwargs)
        for d in decays
    ]


def companion_spectral_radius(operators: Sequence[np.ndarray]) -> float:
    """
    Radio espectral ρ(C) de la matriz companion de bloques asociada a un
    proceso lineal Y_t = Σ_{k=1}^p Ψ_k Y_{t-k}.

    La companion C ∈ ℝ^{pT × pT} es

        C = [ Ψ_1  Ψ_2  ...  Ψ_{p-1}  Ψ_p ]
            [  I    0   ...    0        0  ]
            [  0    I   ...    0        0  ]
            [          ...                ]
            [  0    0   ...    I        0  ]

    El proceso lineal es (débilmente) estacionario sii ρ(C) < 1.

    Para p = 1 coincide con max|λ(Ψ_1)|.
    """
    p = len(operators)
    T = operators[0].shape[0]
    if any(op.shape != (T, T) for op in operators):
        raise ValueError("Todos los operadores deben ser cuadrados y del mismo tamaño.")
    if p == 1:
        return float(np.max(np.abs(np.linalg.eigvals(operators[0]))))
    C = np.zeros((p * T, p * T))
    # Fila de bloques superior: [Ψ_1 ... Ψ_p]
    for k in range(p):
        C[0:T, k * T:(k + 1) * T] = operators[k]
    # Sub-identidades
    for k in range(1, p):
        C[k * T:(k + 1) * T, (k - 1) * T:k * T] = np.eye(T)
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def _op_norm(A: np.ndarray) -> float:
    """Norma espectral (mayor valor singular) de A."""
    return float(np.linalg.norm(A, 2))


# ============================================================
# JERARQUÍA DE MAPAS NO LINEALES F(Y_{t-1}, ..., Y_{t-p})
# ============================================================

class NonlinearFARMap(ABC):
    """
    Interfaz de un mapa dinámico no lineal de orden p.

    Contrato
    --------
    - `p` : int. Número de rezagos que consume el mapa.
    - `n_points` : int. Tamaño de la malla funcional (debe coincidir con los
       operadores y con el dominio del simulador).
    - `predict(lags)` : recibe una lista `lags` de longitud p, donde
       `lags[k]` es Y_{t-(k+1)} (es decir `lags[0]` = Y_{t-1}, el más reciente),
       y devuelve la parte determinística F(...) ∈ ℝ^{n_points}.
    - `stability_report()` : dict de diagnóstico de estacionariedad.
    """

    p: int
    n_points: int

    @abstractmethod
    def predict(self, lags: list[np.ndarray]) -> np.ndarray:
        ...

    @abstractmethod
    def stability_report(self) -> dict:
        ...

    # --- utilidades comunes ---

    def _check_lags(self, lags: list[np.ndarray]) -> None:
        if len(lags) != self.p:
            raise ValueError(f"Se esperaban {self.p} rezagos, se recibieron {len(lags)}.")

    @staticmethod
    def _check_operator_list(ops: Sequence[np.ndarray], p: int, n_points: int, name: str) -> None:
        if len(ops) != p:
            raise ValueError(f"{name}: se esperaban {p} operadores, hay {len(ops)}.")
        for k, op in enumerate(ops):
            if op.shape != (n_points, n_points):
                raise ValueError(
                    f"{name}[{k}] tiene shape {op.shape}; se esperaba "
                    f"({n_points}, {n_points})."
                )

    def __repr__(self) -> str:
        rep = self.stability_report()
        key = "rho_companion_max" if "rho_companion_max" in rep else (
            "contraction_const" if "contraction_const" in rep else "rho_linear_part"
        )
        return (
            f"{self.__class__.__name__}(p={self.p}, "
            f"{key}={rep[key]:.4f}, "
            f"estacionario_probable={rep.get('likely_stationary')})"
        )


# ------------------------------------------------------------
# MÓDULO 1 — Umbral / cambio de régimen (SETAR funcional)
# ------------------------------------------------------------

class ThresholdFAR(NonlinearFARMap):
    """
    FAR(p) de UMBRAL (Self-Exciting Threshold AR funcional).

        régimen r = índice del intervalo de umbrales que contiene z_t,
        z_t = stat( Y_{t-d} )   (variable de transición escalar)
        F(...) = Σ_{k=1}^p Ψ_k^{(r)} Y_{t-k}

    Es no lineal por el cambio discreto de operadores según el régimen, pero
    LINEAL dentro de cada régimen (lo que permite una condición de
    estacionariedad de tipo SETAR vía radio espectral por régimen).

    Parámetros
    ----------
    operators_by_regime : list[list[np.ndarray]]
        Para cada régimen, una lista de p operadores Ψ_k^{(r)}.
        Número de regímenes R = len(operators_by_regime).
    thresholds : secuencia de float
        R-1 umbrales ORDENADOS que parten el soporte de z en R intervalos.
    transition_lag : int, default=1
        d ∈ {1,...,p}: rezago que define la variable de transición.
    stat : Callable, default=media sobre el dominio
        z = stat(Y_{t-d}); por defecto la media de la curva.
    """

    def __init__(
        self,
        operators_by_regime: list[list[np.ndarray]],
        thresholds: Sequence[float],
        transition_lag: int = 1,
        stat: Optional[Callable[[np.ndarray], float]] = None,
    ):
        R = len(operators_by_regime)
        if R < 2:
            raise ValueError("Se requieren >= 2 regímenes para un modelo de umbral.")
        if len(thresholds) != R - 1:
            raise ValueError(f"Se esperaban R-1={R-1} umbrales, hay {len(thresholds)}.")
        if list(thresholds) != sorted(thresholds):
            raise ValueError("`thresholds` debe estar ordenado ascendentemente.")
        p = len(operators_by_regime[0])
        n_points = operators_by_regime[0][0].shape[0]
        for r, ops in enumerate(operators_by_regime):
            self._check_operator_list(ops, p, n_points, f"operators_by_regime[{r}]")
        if not (1 <= transition_lag <= p):
            raise ValueError(f"transition_lag debe estar en [1, {p}].")

        self.p = p
        self.n_points = n_points
        self.operators_by_regime = operators_by_regime
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.transition_lag = transition_lag
        self.stat = stat if stat is not None else (lambda y: float(np.mean(y)))

    def _regime(self, lags: list[np.ndarray]) -> int:
        z = self.stat(lags[self.transition_lag - 1])
        return int(np.searchsorted(self.thresholds, z, side="right"))

    def predict(self, lags: list[np.ndarray]) -> np.ndarray:
        self._check_lags(lags)
        ops = self.operators_by_regime[self._regime(lags)]
        out = np.zeros(self.n_points)
        for k in range(self.p):
            out += ops[k] @ lags[k]
        return out

    def stability_report(self) -> dict:
        radii = [companion_spectral_radius(ops) for ops in self.operators_by_regime]
        rho_max = float(max(radii))
        return {
            "module": "ThresholdFAR",
            "rho_companion_per_regime": radii,
            "rho_companion_max": rho_max,
            "likely_stationary": rho_max < 1.0,
            "note": "Condición SETAR suficiente: max_r rho(C_r) < 1.",
        }


# ------------------------------------------------------------
# MÓDULO 2 — No-linealidad puntual por rezago
# ------------------------------------------------------------

class PointwiseNonlinearFAR(NonlinearFARMap):
    """
    FAR(p) con NO-LINEALIDAD PUNTUAL aplicada a cada rezago antes del operador:

        F(...) = Σ_{k=1}^p Ψ_k g( Y_{t-k} )

    con g aplicada componente a componente. Si g es L_g-Lipschitz, el mapa es
    contracción en el estado apilado (norma del máximo) cuando
    Σ_k L_g·||Ψ_k||_2 < 1 (punto fijo de Banach ⇒ estacionariedad geométrica).

    Parámetros
    ----------
    operators : list[np.ndarray]
        p operadores Ψ_k.
    g : str o Callable, default='tanh'
        No-linealidad puntual. Cadenas disponibles:
          - 'tanh'      : g(y)=tanh(y),      L_g=1     (acotada, segura).
          - 'softsign'  : g(y)=y/(1+|y|),    L_g=1     (acotada, segura).
          - 'sin'       : g(y)=sin(y),       L_g=1.
          - 'cubic_sat' : g(y)=tanh(y^3)/?,  saturada  (ver `g_lipschitz`).
        Si es Callable, DEBE acompañarse de `g_lipschitz`.
    g_lipschitz : float, opcional
        Constante de Lipschitz de g. Obligatoria si g es callable.
        [VERIFICAR EXTERNO] Para g no acotada (p.ej. cuadrática y↦y²) NO existe
        L_g global; en ese caso no hay garantía de estacionariedad y se emite
        advertencia.
    """

    _BUILTINS: dict[str, tuple[Callable, float]] = {
        "tanh":     (np.tanh, 1.0),
        "softsign": (lambda y: y / (1.0 + np.abs(y)), 1.0),
        "sin":      (np.sin, 1.0),
        # cubic saturada: g(y)=tanh(y)^3 ; |g'| = 3 tanh^2 (1-tanh^2) <= 3·(2/3)·(1/3) -> max 4/9
        "cubic_sat": (lambda y: np.tanh(y) ** 3, 4.0 / 9.0),
    }

    def __init__(
        self,
        operators: list[np.ndarray],
        g: Union[str, Callable] = "tanh",
        g_lipschitz: Optional[float] = None,
    ):
        p = len(operators)
        n_points = operators[0].shape[0]
        self._check_operator_list(operators, p, n_points, "operators")
        if isinstance(g, str):
            if g not in self._BUILTINS:
                raise ValueError(f"g='{g}' no reconocida. Opciones: {list(self._BUILTINS)}.")
            self.g, self.g_lipschitz = self._BUILTINS[g]
            self.g_name = g
        else:
            if g_lipschitz is None:
                raise ValueError("Para g callable debe entregar `g_lipschitz`.")
            self.g, self.g_lipschitz = g, float(g_lipschitz)
            self.g_name = getattr(g, "__name__", "callable")

        self.p = p
        self.n_points = n_points
        self.operators = operators

    def predict(self, lags: list[np.ndarray]) -> np.ndarray:
        self._check_lags(lags)
        out = np.zeros(self.n_points)
        for k in range(self.p):
            out += self.operators[k] @ self.g(lags[k])
        return out

    def stability_report(self) -> dict:
        contraction = self.g_lipschitz * sum(_op_norm(op) for op in self.operators)
        return {
            "module": "PointwiseNonlinearFAR",
            "g": self.g_name,
            "g_lipschitz": self.g_lipschitz,
            "contraction_const": float(contraction),
            "likely_stationary": contraction < 1.0,
            "note": "Suficiente (Banach, norma del máximo): L_g·Σ_k||Ψ_k||_2 < 1.",
        }


# ------------------------------------------------------------
# MÓDULO 3 — Bilineal (producto entre rezagos)
# ------------------------------------------------------------

class BilinearFAR(NonlinearFARMap):
    """
    FAR(p) BILINEAL: parte lineal más términos de producto puntual entre rezagos.

        F(...) = Σ_{k=1}^p Ψ_k Y_{t-k}
               + Σ_{(i,j)∈pairs} Φ_{ij} h( Y_{t-i} ⊙ Y_{t-j} )

    donde ⊙ es el producto de Hadamard (puntual) y h es identidad o una
    saturación tanh para acotar el término cuadrático.

    ADVERTENCIA DE ESTABILIDAD
    --------------------------
    Sin saturación (`saturate=False`) el término de producto NO es globalmente
    Lipschitz: el proceso puede DIVERGIR aun con la parte lineal estacionaria.
    Con `saturate=True` el producto pasa por tanh(·/scale) y queda acotado,
    recuperando una dinámica numéricamente estable. Recomendado para simulación.

    Parámetros
    ----------
    operators : list[np.ndarray]
        p operadores lineales Ψ_k.
    bilinear_operators : dict[tuple[int,int], np.ndarray]
        Mapea pares de rezagos (i, j) con 1<=i<=j<=p a su operador Φ_{ij}.
    saturate : bool, default=True
        Si True aplica h(x)=scale·tanh(x/scale) al producto (acotado).
        Si False usa h=identidad (puede divergir).
    scale : float, default=1.0
        Escala de saturación.
    """

    def __init__(
        self,
        operators: list[np.ndarray],
        bilinear_operators: dict[tuple[int, int], np.ndarray],
        saturate: bool = True,
        scale: float = 1.0,
    ):
        p = len(operators)
        n_points = operators[0].shape[0]
        self._check_operator_list(operators, p, n_points, "operators")
        for (i, j), Phi in bilinear_operators.items():
            if not (1 <= i <= j <= p):
                raise ValueError(f"Par bilineal ({i},{j}) inválido; requiere 1<=i<=j<=p.")
            if Phi.shape != (n_points, n_points):
                raise ValueError(f"Φ_{{{i},{j}}} shape {Phi.shape} != ({n_points},{n_points}).")

        self.p = p
        self.n_points = n_points
        self.operators = operators
        self.bilinear_operators = dict(bilinear_operators)
        self.saturate = saturate
        self.scale = float(scale)

    def _h(self, x: np.ndarray) -> np.ndarray:
        if self.saturate:
            return self.scale * np.tanh(x / self.scale)
        return x

    def predict(self, lags: list[np.ndarray]) -> np.ndarray:
        self._check_lags(lags)
        out = np.zeros(self.n_points)
        for k in range(self.p):
            out += self.operators[k] @ lags[k]
        for (i, j), Phi in self.bilinear_operators.items():
            prod = lags[i - 1] * lags[j - 1]          # Hadamard
            out += Phi @ self._h(prod)
        return out

    def stability_report(self) -> dict:
        rho_lin = companion_spectral_radius(self.operators)
        rep = {
            "module": "BilinearFAR",
            "rho_linear_part": rho_lin,
            "saturate": self.saturate,
        }
        if self.saturate:
            # Producto acotado por `scale`; el término bilineal contribuye un
            # sesgo acotado, no afecta la contracción de la parte lineal.
            rep["likely_stationary"] = rho_lin < 1.0
            rep["note"] = ("Con saturación el término bilineal está acotado; "
                           "estabilidad gobernada por rho(parte lineal) < 1.")
        else:
            rep["likely_stationary"] = False
            rep["note"] = ("SIN saturación el término cuadrático no es Lipschitz "
                           "global: posible divergencia. No hay garantía.")
        return rep


# ------------------------------------------------------------
# MÓDULO 4 — Mixtura dependiente del estado (espíritu PSBP)
# ------------------------------------------------------------

class MixtureStateFAR(NonlinearFARMap):
    """
    FAR(p) como MIXTURA dependiente del estado con pesos de tipo Probit
    Stick-Breaking (PSBP), en el espíritu del modelo de la tesis:

        z_t   = stat( Y_{t-d} )                       (estado escalar)
        v_m   = Φ( α_m + β_m · z_t ),  m=1,...,M-1     (Φ = cdf normal estándar)
        w_m   = v_m · Π_{l<m} (1 - v_l),   w_M = Π_{l<M}(1 - v_l)
        F(...) = Σ_{m=1}^M w_m(z_t) · ( Σ_{k=1}^p Ψ_k^{(m)} Y_{t-k} )

    No lineal por la dependencia suave de los pesos en el estado. Es la
    versión "soft" del modelo de umbral y la más cercana a PSBP-FD.

    Parámetros
    ----------
    operators_by_component : list[list[np.ndarray]]
        Para cada componente m, una lista de p operadores Ψ_k^{(m)}.
    alpha, beta : secuencias de float, longitud M-1
        Coeficientes Probit del stick-breaking. β controla la sensibilidad de
        los pesos al estado (β=0 ⇒ pesos constantes ⇒ mixtura no dependiente).
    transition_lag : int, default=1
    stat : Callable, default=media sobre el dominio
    """

    def __init__(
        self,
        operators_by_component: list[list[np.ndarray]],
        alpha: Sequence[float],
        beta: Sequence[float],
        transition_lag: int = 1,
        stat: Optional[Callable[[np.ndarray], float]] = None,
    ):
        M = len(operators_by_component)
        if M < 2:
            raise ValueError("Se requieren >= 2 componentes.")
        if len(alpha) != M - 1 or len(beta) != M - 1:
            raise ValueError(f"alpha y beta deben tener longitud M-1={M-1}.")
        p = len(operators_by_component[0])
        n_points = operators_by_component[0][0].shape[0]
        for m, ops in enumerate(operators_by_component):
            self._check_operator_list(ops, p, n_points, f"operators_by_component[{m}]")
        if not (1 <= transition_lag <= p):
            raise ValueError(f"transition_lag debe estar en [1, {p}].")

        self.p = p
        self.n_points = n_points
        self.M = M
        self.operators_by_component = operators_by_component
        self.alpha = np.asarray(alpha, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.transition_lag = transition_lag
        self.stat = stat if stat is not None else (lambda y: float(np.mean(y)))

    def _weights(self, z: float) -> np.ndarray:
        v = _stdnorm.cdf(self.alpha + self.beta * z)   # (M-1,)
        w = np.empty(self.M)
        remaining = 1.0
        for m in range(self.M - 1):
            w[m] = v[m] * remaining
            remaining *= (1.0 - v[m])
        w[self.M - 1] = remaining
        return w

    def predict(self, lags: list[np.ndarray]) -> np.ndarray:
        self._check_lags(lags)
        z = self.stat(lags[self.transition_lag - 1])
        w = self._weights(z)
        out = np.zeros(self.n_points)
        for m in range(self.M):
            ops = self.operators_by_component[m]
            comp = np.zeros(self.n_points)
            for k in range(self.p):
                comp += ops[k] @ lags[k]
            out += w[m] * comp
        return out

    def stability_report(self) -> dict:
        radii = [companion_spectral_radius(ops) for ops in self.operators_by_component]
        rho_max = float(max(radii))
        return {
            "module": "MixtureStateFAR",
            "rho_companion_per_component": radii,
            "rho_companion_max": rho_max,
            "likely_stationary": rho_max < 1.0,
            "note": ("Suficiente a primer orden: como Σ_m w_m=1, el mapa está "
                     "dominado por max_m rho(C_m) < 1. La dependencia de w en el "
                     "estado añade una corrección Lipschitz típicamente pequeña."),
        }


# ============================================================
# SIMULADOR FAR(p) NO LINEAL
# ============================================================

class NonlinearFARSimulator:
    """
    Simulador funcional FAR(p) no lineal con tendencia determinística.

    Modelo
    ------
        X_t(s) = μ_t(s) + Y_t(s)
        Y_t(s) = F( Y_{t-1}, ..., Y_{t-p} )(s) + ε_t(s)

    con F dado por un `NonlinearFARMap` (uno de los cuatro módulos). El ruido
    ε_t y la tendencia μ_t replican el contrato del módulo lineal.

    Parámetros
    ----------
    domain : FunctionalDomain
    n_curves : int
        T, número de curvas a generar.
    nonlinear_map : NonlinearFARMap
        Mapa dinámico no lineal de orden p. Define `p` automáticamente.
    trend : str o Callable, default='zero'
        Misma semántica que en FARSimulator. Si es str, se delega en
        FARSimulator.TRENDS (con `trend_params` opcional). Si es callable,
        firma (t:int, s:np.ndarray)->np.ndarray.
    trend_params : dict, opcional
    noise_std : float, default=1.0
    noise_type : {'white','smooth'}, default='smooth'
    burn_in : int, default=100
    random_state : int, opcional
    """

    def __init__(
        self,
        domain: FunctionalDomain,
        n_curves: int,
        nonlinear_map: NonlinearFARMap,
        trend: Union[str, Callable] = "zero",
        trend_params: Optional[dict] = None,
        noise_std: float = 1.0,
        noise_type: str = "smooth",
        burn_in: int = 100,
        random_state: Optional[int] = None,
    ):
        if not isinstance(domain, FunctionalDomain):
            raise TypeError("domain debe ser FunctionalDomain.")
        if not isinstance(nonlinear_map, NonlinearFARMap):
            raise TypeError("nonlinear_map debe ser instancia de NonlinearFARMap.")
        if nonlinear_map.n_points != domain.n_points:
            raise ValueError(
                f"nonlinear_map.n_points={nonlinear_map.n_points} != "
                f"domain.n_points={domain.n_points}."
            )
        if noise_std <= 0:
            raise ValueError("noise_std debe ser > 0.")
        if noise_type not in ("white", "smooth"):
            raise ValueError("noise_type debe ser 'white' o 'smooth'.")
        if burn_in < nonlinear_map.p:
            raise ValueError("burn_in debe ser >= p para inicializar los rezagos.")

        self.domain = domain
        self.n_curves = n_curves
        self.map = nonlinear_map
        self.p = nonlinear_map.p
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.burn_in = burn_in
        self.rng = np.random.default_rng(random_state)

        # Resolución de tendencia (reutiliza el catálogo del módulo lineal).
        if isinstance(trend, str):
            if trend not in FARSimulator.TRENDS:
                raise ValueError(
                    f"Tendencia '{trend}' no reconocida. "
                    f"Opciones: {list(FARSimulator.TRENDS)} o un callable."
                )
            if trend_params:
                # Se reusa el builder paramétrico del módulo lineal sin duplicar
                # lógica: se construye un FARSimulator auxiliar mínimo.
                _Psi_dummy = np.zeros((domain.n_points, domain.n_points))
                _aux = FARSimulator(
                    domain=domain, n_curves=n_curves, Psi=_Psi_dummy,
                    trend=trend, trend_params=trend_params,
                    noise_std=noise_std, noise_type=noise_type,
                    burn_in=max(burn_in, self.p), random_state=random_state,
                )
                self.trend_fn = _aux.trend_fn
            else:
                _base = FARSimulator.TRENDS[trend]
                self.trend_fn = lambda t, s: _base(t, s, self.n_curves)
        else:
            self.trend_fn = trend

        # Diagnóstico de estabilidad del mapa.
        self.stability_ = self.map.stability_report()
        if not self.stability_.get("likely_stationary", True):
            warnings.warn(
                f"El mapa {self.map.__class__.__name__} no satisface la condición "
                f"de estacionariedad suficiente: {self.stability_}. "
                "El proceso puede ser no estacionario o divergir.",
                UserWarning,
                stacklevel=2,
            )

        self.data_: Optional[np.ndarray] = None

    # ---- Ruido (idéntico al módulo lineal) ----

    def _draw_noise(self) -> np.ndarray:
        z = self.rng.normal(loc=0.0, scale=self.noise_std, size=self.domain.n_points)
        if self.noise_type == "white":
            return z
        z_s = gaussian_filter1d(z, sigma=2.0)
        std_s = z_s.std()
        if std_s > 1e-12:
            z_s *= self.noise_std / std_s
        return z_s

    # ---- Simulación principal ----

    def simulate(self) -> np.ndarray:
        """
        Genera X ∈ ℝ^{n_curves × n_points}. Mantiene una ventana `history` de los
        últimos p estados latentes Y, con history[0] = Y_{t-1} (más reciente).
        """
        T = self.n_curves
        npts = self.domain.n_points
        X = np.zeros((T, npts))

        # Inicialización de los p rezagos con ruido.
        history = [self._draw_noise() for _ in range(self.p)]  # history[0]=más reciente

        # Burn-in.
        for _ in range(self.burn_in):
            eps = self._draw_noise()
            y_new = self.map.predict(history) + eps
            history = [y_new] + history[:-1]

        # Simulación.
        for t in range(T):
            eps_t = self._draw_noise()
            y_t = self.map.predict(history) + eps_t
            mu_t = self.trend_fn(t, self.domain.grid)
            X[t, :] = mu_t + y_t
            history = [y_t] + history[:-1]

        self.data_ = X
        return X

    # ---- Diagnóstico ----

    def summary_stats(self, n_lags: int = 10) -> dict:
        """Métricas de validación del proceso (requiere .simulate() previo)."""
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        X = self.data_
        mean_fn = X.mean(axis=0)
        std_fn = X.std(axis=0)

        idx = np.argmin(np.abs(self.domain.grid - 0.5))
        series = X[:, idx]
        acf = [float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
               for lag in range(1, n_lags + 1)]
        _, p_norm = normaltest(series)

        Cov = np.cov(X.T)
        max_eig_cov = float(np.linalg.eigvalsh(Cov).max())

        return {
            "p_order": self.p,
            "map": self.map.__class__.__name__,
            "mean_fn_max_abs": float(np.abs(mean_fn).max()),
            "std_fn_mean": float(std_fn.mean()),
            "acf": acf,
            "acf_lag1": acf[0],
            "acf_lag2": acf[1] if len(acf) > 1 else float("nan"),
            "normality_p_at_s05": float(p_norm),
            "max_eigenvalue_cov_empirical": max_eig_cov,
            "stability": self.stability_,
            "finite": bool(np.all(np.isfinite(X))),
        }

    # ---- Visualización (paralela al módulo lineal) ----

    def plot_curves(self, n_show: int = 50, alpha: float = 0.3,
                    title: Optional[str] = None, figsize=(10, 5)) -> None:
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        fig, ax = plt.subplots(figsize=figsize)
        for t in range(min(n_show, self.n_curves)):
            ax.plot(self.domain.grid, self.data_[t], alpha=alpha, color="seagreen", lw=0.8)
        ax.plot(self.domain.grid, self.data_.mean(axis=0),
                color="crimson", lw=2.0, label="Media empírica")
        ax.set_xlabel("s (dominio funcional)")
        ax.set_ylabel("X_t(s)")
        ax.set_title(title or f"FAR({self.p}) no lineal — {self.map.__class__.__name__}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, title: Optional[str] = None, figsize=(10, 6)) -> None:
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(
            self.data_, aspect="auto", cmap="RdBu_r",
            extent=[self.domain.grid.min(), self.domain.grid.max(), self.n_curves, 0],
            vmin=np.percentile(self.data_, 1), vmax=np.percentile(self.data_, 99),
        )
        ax.set_xlabel("s")
        ax.set_ylabel("t (índice de curva)")
        ax.set_title(title or f"Heatmap FAR({self.p}) no lineal")
        plt.colorbar(im, ax=ax, label="X_t(s)")
        plt.tight_layout()
        plt.show()

    def plot_acf(self, s_points: Optional[list] = None, n_lags: int = 20,
                 title: Optional[str] = None, figsize=(10, 4)) -> None:
        if self.data_ is None:
            raise RuntimeError("Ejecuta .simulate() primero.")
        s_points = s_points or [0.2, 0.5, 0.8]
        fig, ax = plt.subplots(figsize=figsize)
        for s_val in s_points:
            idx = np.argmin(np.abs(self.domain.grid - s_val))
            series = self.data_[:, idx]
            acf = [1.0] + [
                float(np.corrcoef(series[:-lag], series[lag:])[0, 1])
                for lag in range(1, n_lags + 1)
            ]
            ax.plot(range(len(acf)), acf, marker="o", markersize=3, label=f"s={s_val}")
        ci = 2 / np.sqrt(self.n_curves)
        ax.axhline(ci, color="gray", ls="--", lw=0.8, label=f"±2/√T = ±{ci:.2f}")
        ax.axhline(-ci, color="gray", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(title or "Autocorrelación funcional (puntos s fijos)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(domain={self.domain.n_points} pts, "
            f"n_curves={self.n_curves}, p={self.p}, map={self.map.__class__.__name__}, "
            f"estacionario_probable={self.stability_.get('likely_stationary')})"
        )


# ============================================================
# EJEMPLO DE USO — los cuatro módulos con p arbitrario
# ============================================================

if __name__ == "__main__":
    domain = FunctionalDomain.regular(s_min=0, s_max=1, n_points=80)
    p = 2  # orden parametrizable: cámbielo libremente

    # Operadores base por rezago (decaimiento geométrico => estacionariedad).
    base_ops = build_lag_operators(domain, p=p, bandwidth=0.05,
                                   base_decay=0.5, decay_ratio=0.5)

    # --- Módulo 1: Umbral (2 regímenes) ---
    ops_reg2 = build_lag_operators(domain, p=p, bandwidth=0.05,
                                   base_decay=0.6, decay_ratio=0.5)
    m1 = ThresholdFAR(operators_by_regime=[base_ops, ops_reg2],
                      thresholds=[0.0], transition_lag=1)

    # --- Módulo 2: No-linealidad puntual (tanh) ---
    m2 = PointwiseNonlinearFAR(operators=base_ops, g="tanh")

    # --- Módulo 3: Bilineal (producto Y_{t-1} ⊙ Y_{t-2}), saturado ---
    Phi = 0.4 * build_integral_matrix(domain, gaussian_integral_kernel,
                                      bandwidth=0.05, decay=1.0)
    m3 = BilinearFAR(operators=base_ops,
                     bilinear_operators={(1, 2): Phi},
                     saturate=True, scale=1.0)

    # --- Módulo 4: Mixtura PSBP (2 componentes) ---
    ops_comp2 = build_lag_operators(domain, p=p, bandwidth=0.05,
                                    base_decay=0.6, decay_ratio=0.5)
    m4 = MixtureStateFAR(operators_by_component=[base_ops, ops_comp2],
                         alpha=[0.0], beta=[2.0], transition_lag=1)

    for tag, mp in [("Umbral", m1), ("Puntual", m2), ("Bilineal", m3), ("Mixtura", m4)]:
        sim = NonlinearFARSimulator(
            domain=domain, n_curves=300, nonlinear_map=mp,
            trend="zero", noise_std=0.5, noise_type="smooth",
            burn_in=120, random_state=42,
        )
        X = sim.simulate()
        st = sim.summary_stats()
        print(f"[{tag:8s}] {mp}")
        print(f"           finito={st['finite']}  "
              f"acf1={st['acf_lag1']:+.3f}  acf2={st['acf_lag2']:+.3f}  "
              f"std_fn={st['std_fn_mean']:.3f}")
