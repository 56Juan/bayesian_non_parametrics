"""
sim_escenario_B1.py
====================
Algoritmo B-1 del anexo (`docs/01 Anexo.tex`, `ane_00_02_01_alg_b1`): mezcla de
K = 3 mecanismos funcionales LINEALES, con la variable de asignacion Z_t
independiente de la curva anterior.

Modelo generador (Cuadro `tab:ane_algB1`)
------------------------------------------
    X_t(tau) = mu_{Z_t}(tau) + sum_{j=1}^{3} beta_{Z_t,j}(tau) R_{t-1,j} + eps_t(tau)

    R_{t-1,j} = int_0^1 X_{t-1}(s) w_j(s) ds,      j = 1, 2, 3,
    P(Z_t = k) = pi_k = 1/3,   Z_t independiente de X_{t-1} y entre periodos.

Los rasgos funcionales w_j = phi_j
-----------------------------------
El anexo deja w_j(s) "prefijadas"; se eligen los tres primeros polinomios de
Legendre desplazados a [0, 1], ORTONORMALES en L^2([0, 1]):

    phi_1(tau) = 1                          (nivel)
    phi_2(tau) = sqrt(3) (2 tau - 1)         (pendiente)
    phi_3(tau) = sqrt(5) (6 tau^2 - 6 tau + 1)   (curvatura)

Da tres rasgos interpretables (nivel, pendiente, curvatura de la curva
anterior) y, junto con la eleccion de `beta_{kj}(tau) = b_{kj} phi_j(tau)`
--el coeficiente del mecanismo k sobre el rasgo j se toma proporcional al
MISMO modo phi_j--, hace que la dinamica de los tres rasgos quede DESACOPLADA:
por ortonormalidad (<phi_j', phi_j> = delta_{j'j}),

    R_{t,j} = <phi_j, mu_{Z_t}> + b_{Z_t,j} R_{t-1,j} + <phi_j, eps_t>,

de modo que cada rasgo j es, dado el historial de mecanismos, un AR(1) escalar
de coeficiente aleatorio (b_{Z_t,j} segun el mecanismo sorteado en t), sin
acoplarse a los otros dos rasgos. La condicion de estacionariedad de un AR(1)
de coeficiente aleatorio iid es E[b_{Z,j}^2] < 1; con `B_MECANISMOS_B1` y
pi_k = 1/3 el maximo de E[b_{Z,j}^2] sobre los tres rasgos es ~0.354, lejos
del limite (se eligio deliberadamente alto, no el minimo que garantiza
estacionariedad, para que la fraccion de la curva que SI depende linealmente
de R_{t-1} sea apreciable frente a la incertidumbre irreducible del sorteo de
Z_t; ver `B_MECANISMOS_B1`). Verificado por simulacion directa del proceso de
rasgos (4000 pasos, 50 trayectorias): |R| se mantiene acotado (< 2.5) y la
varianza por bloques de 500 pasos es estable, sin tendencia.

Esta eleccion (beta_{kj} propocional a phi_j) es una decision de diseno, no
una exigencia del anexo; se documenta aqui porque es la que garantiza
estacionariedad de forma analitica en vez de por ensayo y error. B-2 y B-3
(que heredan `funciones_rasgo`, `medias_mezcla` y el motor `simular_mezcla_
curvas` de este modulo) abandonan la estructura diagonal al introducir
terminos cuadraticos y de interaccion, y se calibran por separado.

Las medias mu_k
----------------
Tres formas distintas pero DELIBERADAMENTE CERCANAS entre si (`media_seno` es
el `mu(tau) = sin(2 pi tau)` que ya usan A-1/A-2/A-3 y C-1..C-3; aqui es
ademas `mu_1`):

    mu_1(tau) = sin(2 pi tau)
    mu_2(tau) = sin(2 pi tau) + 0.3 sin(4 pi tau)
    mu_3(tau) = 0.85 sin(2 pi tau) - 0.2

Que tan distintas se hacen mu_1, mu_2, mu_3 no es un detalle cosmetico: el
sorteo de Z_t es INDEPENDIENTE de X_{t-1} (en B-1 y B-2), de modo que la
varianza de "que media tocó" es enteramente IRREDUCIBLE para cualquier
predictor que solo observe la curva anterior, oraculo incluido. Si mu_1, mu_2,
mu_3 fueran muy distintas entre si (una prueba inicial uso una diferencia de
fase y una reescala 1.3x: r2_oraculo_1rezago cayo a ~0.07-0.18 segun el
algoritmo), esa varianza de sorteo domina la varianza total de X_t y el
oraculo de un rezago queda cerca de cero SIN IMPORTAR que tan fuerte sea la
retroalimentacion beta_{kj}: hubiera dejado a los tres algoritmos B sin nada
que ningun metodo, ni siquiera el oraculo, pueda predecir mejor que la media
incondicional --lo opuesto al foco del estudio (CLAUDE.md §2: "escenarios
donde el modelo puede ganar"). Con las medias CERCANAS de arriba y los
coeficientes ALTOS de `B_MECANISMOS_B1` (ver mas abajo), r2_oraculo_1rezago de
B-1 sube a ~0.50 (T = 1000, R = 5, sigma_obs = 0.25): los tres mecanismos son
lineales, de modo que la retroalimentacion domina la varianza de sorteo de
Z_t.

B-2 y B-3 (`sim_escenario_B2.py`, `sim_escenario_B3.py`) quedan con
r2_oraculo_1rezago mas bajo (~0.13 y ~0.08 respectivamente, mismos T, R,
sigma_obs), y NO por falta de calibracion: sus mecanismos 2 y 3 son
cuadraticos/de interaccion (R1^2, R2 R3, R1 R2, ...), y con R_{t-1} de escala
moderada (|R| tipicamente < 2, ver `sim_escenario_B2`) el termino cuadratico
Var(R^2) ~ 2 Var(R)^2 es intrinsecamente chico frente al termino lineal
Var(R): subir su coeficiente para igualar el R^2 de B-1 exigiria un
coeficiente varias veces mayor, y con un termino que crece con el CUADRADO de
R eso arriesga la estacionariedad (ver docstring de `sim_escenario_B2`). Es el
mismo patron deliberado que ya usa el Algoritmo C-1 del anexo --dependencia
lineal debil (c = 0.15, c^2 = 0.0225) y dependencia cuadratica mas fuerte
(b = 0.5) que un FAR gaussiano no puede aprovechar--: B-2 y B-3 no buscan
maximizar r2_oraculo_1rezago sino dejar una señal no lineal genuina que un
metodo lineal no capture, aunque el techo agregado sea mas bajo que en B-1.

La innovacion eps_t(tau) es la misma innovacion funcional gaussiana de
covarianza exponencial cuadratica que usan A-1/B-1(historico)/B-2/B-3
(`matriz_covarianza_innovacion`), con `sigma_eps = 0.35`, `ell = 0.25`:
calibrados (ver docstring de `K_MAT` mas abajo) para que la varianza inducida
sobre el rasgo de nivel sea la mayor de las tres y la de curvatura la menor,
sin ser nula.

El oraculo de un rezago
------------------------
Como Z_t es independiente de X_{t-1} (B-1 y B-2; no B-3, ver su modulo), el
oraculo de Bayes de un rezago es exacto y en forma cerrada, la MEZCLA sobre
los tres mecanismos:

    E[X_t | X_{t-1}] = sum_k pi_k (mu_k(tau) + sum_j beta_{kj}(tau) R_{t-1,j}).

Se guarda en `internos["media_condicional"]`. La ley condicional de X_t dado
X_{t-1} es una mezcla de 3 componentes gaussianas (dado R_{t-1} fijo, cada
componente es X_{t-1}-condicionalmente gaussiana via eps_t): es exactamente el
escenario donde la mezcla probit del PSBPM-FD tiene algo real que un FAR
gaussiano de un solo componente no puede representar.

`Z_t` NO se entrega a la estimacion: vive en `internos["mecanismo"]` y solo
sirve de control de calidad (junto con `internos["rasgos"]`, los R_{t-1}
usados para generar cada curva retenida).

Contrato de reproducibilidad
------------------------------
Por replica se abre UN generador `default_rng(hija)`. En cada paso del
calentamiento y de las curvas retenidas se consume, en este orden: los rasgos
NO consumen aleatoriedad (son una funcion determinista de la curva anterior),
el sorteo de Z_i (`rng.choice`), y la innovacion eps_i (`rng.standard_normal`
via el factor de Cholesky). La curva inicial X_0 (antes de i = 0) se extrae
directamente de la innovacion, un arranque razonable que el calentamiento
olvida.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    aplicar_ruido_observacion,
    diagnostico_comun,
    factor_cholesky,
    generador_innovacion,
    grilla_regular,
    matriz_covarianza_innovacion,
    pesos_trapezoidales,
    semillas_replicas,
)
from .sim_series_clasicas import r2_empirico_media_condicional

__all__ = [
    "media_seno",
    "media_mecanismo_2",
    "media_mecanismo_3",
    "medias_mezcla",
    "funciones_rasgo",
    "calcular_rasgos",
    "transformar_identidad",
    "B_MECANISMOS_B1",
    "ConfigEscenarioB1",
    "simular_mezcla_curvas",
    "generar_escenario_B1",
    "resumen_escenario_B1",
]


# ==========================================================================
# RASGOS FUNCIONALES Y MEDIAS DE LOS MECANISMOS
# ==========================================================================

def funciones_rasgo(tau: np.ndarray) -> np.ndarray:
    """
    Phi (3, L): los tres pesos w_j = phi_j evaluados en la grilla, los
    polinomios de Legendre desplazados a [0, 1] de grado 0, 1 y 2,
    ortonormales en L^2([0, 1]) (ver docstring del modulo).
    """
    tau = np.asarray(tau, dtype=float)
    phi1 = np.ones_like(tau)
    phi2 = np.sqrt(3.0) * (2.0 * tau - 1.0)
    phi3 = np.sqrt(5.0) * (6.0 * tau ** 2 - 6.0 * tau + 1.0)
    return np.stack([phi1, phi2, phi3])


def calcular_rasgos(X: np.ndarray, pesos: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """
    R_j = int X(s) phi_j(s) ds ~= sum_l X_l phi_j_l w_l, por cuadratura
    trapezoidal. X puede ser (L,) o (..., L); retorna (3,) o (..., 3).
    """
    Phi_w = Phi * pesos[None, :]          # (3, L)
    return X @ Phi_w.T


def media_seno(tau: np.ndarray) -> np.ndarray:
    """mu_1(tau) = sin(2 pi tau): la media que ya comparten A-1..A-3 y C-1..C-3."""
    return np.sin(2.0 * np.pi * np.asarray(tau, dtype=float))


def media_mecanismo_2(tau: np.ndarray) -> np.ndarray:
    """mu_2(tau) = sin(2 pi tau) + 0.3 sin(4 pi tau): mu_1 mas un armonico chico."""
    tau = np.asarray(tau, dtype=float)
    return np.sin(2.0 * np.pi * tau) + 0.3 * np.sin(4.0 * np.pi * tau)


def media_mecanismo_3(tau: np.ndarray) -> np.ndarray:
    """mu_3(tau) = 0.85 sin(2 pi tau) - 0.2: mu_1 amortiguada mas un desplazamiento chico."""
    tau = np.asarray(tau, dtype=float)
    return 0.85 * np.sin(2.0 * np.pi * tau) - 0.2


def medias_mezcla(tau: np.ndarray) -> np.ndarray:
    """MU (3, L): mu_1, mu_2, mu_3 apiladas."""
    return np.stack([media_seno(tau), media_mecanismo_2(tau), media_mecanismo_3(tau)])


# ==========================================================================
# MECANISMO B-1 (LINEAL)
# ==========================================================================

def transformar_identidad(R: np.ndarray, k: int) -> np.ndarray:
    """z_k(R) = R para los tres mecanismos: la version LINEAL de B-1."""
    return R


B_MECANISMOS_B1 = np.array([
    [0.85, 0.40, 0.15],
    [0.30, 0.80, 0.35],
    [-0.50, 0.20, 0.75],
])
"""
Matriz (mecanismo x rasgo) de `beta_{kj}(tau) = B_MECANISMOS_B1[k, j] phi_j(tau)`.
Heterogeneidad entre mecanismos (Cuadro tab:ane_algB1: "la heterogeneidad se
encuentra en los parametros de generacion de cada mecanismo"): el mecanismo 1
responde sobre todo al nivel, el 2 a la pendiente, el 3 (con signo negativo en
nivel) a la curvatura. `E[b_{Z,j}^2]` maximo sobre j es ~0.354 con pi = 1/3,
lejos del limite de 1 (ver docstring del modulo para la condicion de
estacionariedad); se eligio deliberadamente ALTO (no el minimo que garantiza
estacionariedad) para que la parte de la curva que SI depende linealmente de
R_{t-1} sea una fraccion apreciable de la varianza total y el oraculo de un
rezago no quede dominado por la incertidumbre irreducible del sorteo de Z_t.
"""


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioB1(ConfigObservacion):
    """
    Algoritmo B-1: mezcla de 3 mecanismos lineales, asignacion independiente.

    pi : probabilidades de los 3 mecanismos, deben sumar 1. (1/3, 1/3, 1/3).
    sigma_eps, ell : escala y longitud de correlacion de la innovacion
        funcional gaussiana eps_t(tau) (nucleo exponencial cuadratico).
    prop_train_referencia : solo para el diagnostico; el corte T0 real lo fija
        el notebook.
    """

    pi: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    sigma_eps: float = 0.35
    ell: float = 0.25
    prop_train_referencia: float = 0.70

    def validar(self) -> None:
        super().validar()
        if len(self.pi) != 3:
            raise ValueError("pi debe tener exactamente 3 componentes.")
        if any(p < 0 for p in self.pi):
            raise ValueError("pi no puede tener componentes negativas.")
        if abs(sum(self.pi) - 1.0) > 1e-8:
            raise ValueError(f"pi debe sumar 1; suma {sum(self.pi)}.")
        if self.sigma_eps <= 0:
            raise ValueError("sigma_eps debe ser positivo.")
        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")
        if not 0.0 < self.prop_train_referencia < 1.0:
            raise ValueError("prop_train_referencia debe estar en (0, 1).")


# ==========================================================================
# MOTOR COMUN A B-1, B-2 Y B-3
# ==========================================================================

def simular_mezcla_curvas(
    cfg: ConfigObservacion,
    Phi: np.ndarray,
    pesos: np.ndarray,
    MU: np.ndarray,
    B: np.ndarray,
    transformar_rasgos: Callable[[np.ndarray, int], np.ndarray],
    pi_efectiva: Callable[[np.ndarray], np.ndarray],
    chol_K: np.ndarray,
    burn_in: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Motor comun a los tres algoritmos de mezcla (B-1, B-2, B-3): itera

        R_{i-1} = calcular_rasgos(X_{i-1}),
        p_i = pi_efectiva(R_{i-1}),          Z_i ~ Categorical(p_i),
        X_i = mu_{Z_i} + (B[Z_i] * transformar_rasgos(R_{i-1}, Z_i)) @ Phi + eps_i,

    y devuelve, para las T curvas retenidas (i >= burn_in):

        curvas (T, L), Z (T,) el mecanismo sorteado, R (T, 3) los rasgos de la
        curva ANTERIOR usados para generarla, y `oraculo` (T, L) la mezcla de
        Bayes sum_k p_i[k] (mu_k + contrib_k(R_{i-1})) --exacta porque p_i solo
        depende de R_{i-1}, observado--.

    `pi_efectiva` ignora su argumento en B-1/B-2 (pi fija) y lo usa en B-3
    (softmax dependiente de X_{t-1}); `transformar_rasgos` es la identidad en
    B-1 y depende del mecanismo en B-2/B-3. X_0 se extrae de la innovacion.
    """
    n_pasos = int(burn_in) + int(cfg.T)
    L = Phi.shape[1]
    innovacion = generador_innovacion(chol_K, rng)

    X_prev = innovacion()
    curvas = np.empty((int(cfg.T), L))
    Z_hist = np.empty(int(cfg.T), dtype=int)
    R_hist = np.empty((int(cfg.T), 3))
    oraculo = np.empty((int(cfg.T), L))

    for i in range(n_pasos):
        R_prev = calcular_rasgos(X_prev, pesos, Phi)
        p = np.asarray(pi_efectiva(R_prev), dtype=float)
        k = int(rng.choice(3, p=p))
        contrib_k = (B[k] * transformar_rasgos(R_prev, k)) @ Phi
        X = MU[k] + contrib_k + innovacion()
        if i >= burn_in:
            t = i - burn_in
            curvas[t] = X
            Z_hist[t] = k
            R_hist[t] = R_prev
            contrib_todos = np.stack([
                (B[kk] * transformar_rasgos(R_prev, kk)) @ Phi for kk in range(3)
            ])
            oraculo[t] = (p[:, None] * (MU + contrib_todos)).sum(axis=0)
        X_prev = X
    return curvas, Z_hist, R_hist, oraculo


def _pi_fija(pi: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Devuelve el callable `pi_efectiva` constante que espera el motor."""
    def pi_efectiva(R_prev: np.ndarray) -> np.ndarray:
        return pi
    return pi_efectiva


# ==========================================================================
# GENERADOR B-1
# ==========================================================================

def generar_escenario_B1(cfg: ConfigEscenarioB1,
                         diagnosticar: bool = True) -> SalidaSimulacion:
    """
    Genera R replicas del Algoritmo B-1: mezcla de 3 mecanismos lineales con
    asignacion independiente de la curva anterior. La grilla, los rasgos, las
    medias y la factorizacion de la innovacion se construyen una sola vez y se
    comparten entre replicas.
    """
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    pesos = pesos_trapezoidales(tau)
    Phi = funciones_rasgo(tau)
    MU = medias_mezcla(tau)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)
    pi = np.asarray(cfg.pi, dtype=float)
    pi_efectiva = _pi_fija(pi)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)
    R_, T, L = int(cfg.R), int(cfg.T), int(cfg.L)
    observaciones = np.empty((R_, T, L))
    curvas = np.empty((R_, T, L))
    mc = np.empty((R_, T, L))
    Z_all = np.empty((R_, T), dtype=int)
    Rasgos_all = np.empty((R_, T, 3))

    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        curvas_r, Z_r, R_r, mc_r = simular_mezcla_curvas(
            cfg, Phi, pesos, MU, B_MECANISMOS_B1, transformar_identidad,
            pi_efectiva, chol_K, int(cfg.burn_in), rng,
        )
        curvas[r] = curvas_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)
        mc[r] = mc_r
        Z_all[r] = Z_r
        Rasgos_all[r] = R_r

    media = pi @ MU
    internos = {
        "funciones_rasgo": Phi,
        "pesos_cuadratura": pesos,
        "medias_mecanismos": MU,
        "matriz_coeficientes": B_MECANISMOS_B1,
        "cov_innovacion": K,
        "pi": pi,
        "mecanismo": Z_all,
        "rasgos": Rasgos_all,
        "media_condicional": mc,
    }
    salida = SalidaSimulacion(
        observaciones=observaciones, curvas=curvas, grilla=tau, media=media,
        semillas=registro, config=cfg, internos=internos,
    )
    if diagnosticar:
        salida.diagnostico = resumen_escenario_B1(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD
# ==========================================================================

def resumen_escenario_B1(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del B-1: el diagnostico comun, la frecuencia empirica
    de cada mecanismo contra pi (deben coincidir dentro del error de Monte
    Carlo), el maximo absoluto de los rasgos (acotamiento, ver docstring del
    modulo) y el R^2 del oraculo de mezcla de un rezago.
    """
    if not isinstance(salida.config, ConfigEscenarioB1):
        raise TypeError("resumen_escenario_B1 requiere ConfigEscenarioB1; se "
                        f"recibio {type(salida.config).__name__}.")
    cfg = salida.config
    Z = salida.internos["mecanismo"]
    Rasgos = salida.internos["rasgos"]
    pi = salida.internos["pi"]
    frac = np.array([float((Z == k).mean()) for k in range(3)])

    return {
        **diagnostico_comun(salida),
        "pi_objetivo": pi.tolist(),
        "pi_empirica": frac.tolist(),
        "pi_error_absoluto_max": float(np.max(np.abs(frac - pi))),
        "max_abs_rasgos": float(np.max(np.abs(Rasgos))),
        "var_rasgos": Rasgos.reshape(-1, 3).var(axis=0).tolist(),
        "sigma_eps": float(cfg.sigma_eps),
        "ell": float(cfg.ell),
        "r2_oraculo_1rezago": r2_empirico_media_condicional(salida),
        "r2_oraculo_1rezago_empirico": r2_empirico_media_condicional(salida),
        "T0_referencia": int(np.floor(cfg.prop_train_referencia * cfg.T)),
    }
