"""
sim_escenario_4.py
==================
Escenario 4: proceso autorregresivo funcional lineal y homogeneo con
innovaciones de mezcla de escala skew-normal (Algoritmo 4 del anexo).

Modelo generador
----------------
    X_t(tau) = mu(tau) + int_0^1 psi(tau, s) (X_{t-1}(s) - mu(s)) ds + eps_t(tau)

    eps_t = W_t^{-1/2} ( Delta |U_{0t}| + Gamma^{1/2} U_{1t} ) - k_1 sqrt(2/pi) Delta,

con U_{0t} ~ N(0, 1) escalar, U_{1t} ~ N_L(0, I) y W_t un factor de escala
positivo independiente de ambos. Esta es la representacion estocastica de la
familia de mezclas de escala skew-normal (SMSN) de Lachos, Bandyopadhyay y
Garay (2011), en la que la eleccion de la ley de W determina el miembro:

    "sn"   W = 1                      skew-normal
    "st"   W ~ Gamma(nu/2, nu/2)      skew-t
    "sl"   W ~ Beta(nu, 1)            skew-slash
    "scn"  W = escala con prob. nu,   skew-normal contaminada
           W = 1 con prob. 1 - nu

La sustraccion de k_1 sqrt(2/pi) Delta, con k_1 = E[W^{-1/2}], centra la
innovacion, de modo que la funcion media del proceso sigue siendo mu.

Proposito del escenario
-----------------------
La dinamica es lineal y homogenea en el tiempo, identica a la del Escenario 1;
lo unico que cambia es la ley de la innovacion. El escenario aisla, en
consecuencia, el tercer supuesto estructural de los modelos FAR, FMA y FARMA:
la rigidez distribucional. Bajo este generador la ley condicional de la curva
dada su historia es asimetrica y, segun el miembro de la familia, de colas
pesadas, pero conserva una media condicional lineal correctamente
especificada. Un metodo que solo estime la media condicional no sufrira
penalizacion apreciable en error puntual, mientras que las medidas de calidad
distribucional deberian separar a los metodos con verosimilitud gaussiana de
aquellos que estiman la densidad completa.

Calibracion de la escala
------------------------
Los parametros de asimetria y de cola alteran la varianza de la innovacion, de
modo que compararlos entre configuraciones sin corregir confundiria el efecto
distribucional con un cambio de razon senal-ruido. Para evitarlo, la asimetria
se parametriza mediante el coeficiente

    delta_l = Delta_l / sqrt( Delta_l^2 + Gamma_ll )  en  (-1, 1),

que es el parametro de forma estandar de la familia skew-normal: adimensional,
independiente de la escala y ---a diferencia de la magnitud bruta de Delta---
independiente tambien de la finura de la grilla y de la longitud de
correlacion. Fijado delta y la varianza objetivo sigma_eps^2, los componentes
quedan determinados por

    Delta_l   = sigma_eps * delta_l / sqrt( coef * delta_l^2 + k_2 (1 - delta_l^2) ),
    Gamma_ll  = sigma_eps^2 * (1 - delta_l^2) / ( coef * delta_l^2 + k_2 (1 - delta_l^2) ),
    Gamma     = D C D,   D = diag( sqrt(Gamma_ll) ),

con coef = k_2 - 2 k_1^2 / pi, k_1 = E[W^{-1/2}], k_2 = E[W^{-1}] y C la matriz
de correlacion exponencial cuadratica de longitud ell. De esta construccion se
sigue Var(eps_l) = coef Delta_l^2 + k_2 Gamma_ll = sigma_eps^2 de forma exacta,
para cualquier delta, cualquier miembro de la familia y cualquier valor de nu.
Gamma es ademas semidefinida positiva por construccion, al ser un reescalado
diagonal de una matriz de correlacion, de modo que la especificacion no admite
combinaciones invalidas de parametros.

Cabe precisar que la componente asimetrica es un choque comun a todo el
dominio, pues U_0 es escalar: la innovacion es la suma de un desplazamiento
asimetrico de la curva completa y de un campo gaussiano suave. En consecuencia
la covarianza resultante, coef Delta Delta' + k_2 Gamma, presenta correlacion
de mayor alcance que la exponencial cuadratica del Escenario 1. Esto no es un
artefacto de la calibracion sino una consecuencia necesaria de que exista un
choque comun; la varianza puntual, que es lo que gobierna la razon senal-ruido,
permanece igualada. Una asimetria localizada en una region del dominio se
obtiene mediante `forma_skew_fn`.

Asimetria de la innovacion y asimetria de la curva
--------------------------------------------------
La curva es una suma ponderada de innovaciones pasadas, X_t - mu = sum_k Psi^k
eps_{t-k}, de modo que su asimetria puntual no coincide con la de la innovacion
que la genera. La direccion del efecto depende de como actue el operador sobre
cada componente: la agregacion atenua la asimetria en la medida en que el
operador trate por igual al choque asimetrico y al campo gaussiano, pero puede
amplificarla cuando el operador suaviza mas el campo gaussiano de corta
correlacion que el choque comun, que por ser constante sobre el dominio
sobrevive al suavizado. El diagnostico reporta ambas asimetrias y su razon, de
modo que el efecto se verifica para cada configuracion en lugar de suponerse.
En cualquier caso, una asimetria de curva proxima a cero indica que el
escenario no discriminara entre metodos y aconseja revisar la combinacion de
delta_skew, hs_norm y ell.

El esquema de observacion, la cuadratura del operador y el control de calidad
transversal provienen de `sim_comun.py`; este modulo aporta unicamente la ley
de la innovacion y sus verificaciones especificas.

Uso tipico desde un notebook
----------------------------
    from model_psbp_fd.pipelines import (
        ConfigEscenario4, generar_escenario_4, guardar_escenario
    )

    cfg = ConfigEscenario4(L=48, T=200, burn_in=200, R=50, seed=20260719,
                           gamma=0.3, hs_norm=0.5, sigma_eps=1.0, ell=0.2,
                           familia="st", delta_skew=0.85, nu=5.0,
                           sigma_obs=0.5)
    salida = generar_escenario_4(cfg)
    salida.diagnostico            # control de calidad del generador
    X = salida.observaciones      # (R, T, L) -> insumo del pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import gammaln

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    grilla_regular,
    evaluar_media,
    matriz_operador_ar,
    matriz_covarianza_innovacion,
    factor_cholesky,
    semillas_replicas,
    aplicar_ruido_observacion,
    diagnostico_comun,
    norma_hilbert_schmidt,
    pesos_trapezoidales,
)

__all__ = [
    "ConfigEscenario4",
    "generar_escenario_4",
    "resumen_escenario_4",
    "simular_trayectoria_far_smsn",
    "momentos_mezcla_escala",
    "extraer_factor_escala",
    "construir_componentes_smsn",
    "generador_innovacion_smsn",
]

_FAMILIAS = ("sn", "st", "sl", "scn")
_SQRT_2_PI = float(np.sqrt(2.0 / np.pi))


# ==========================================================================
# FAMILIA DE MEZCLAS DE ESCALA
# ==========================================================================

def momentos_mezcla_escala(
    familia: str, nu: float, escala_contaminacion: float
) -> tuple[float, float]:
    """
    Momentos inversos del factor de escala:

        k_1 = E[W^{-1/2}],   k_2 = E[W^{-1}].

    El primero interviene en el centrado de la innovacion y el segundo en su
    varianza, de modo que ambos son necesarios para calibrar la escala. Sus
    condiciones de existencia acotan el rango admisible de nu: la varianza de
    la innovacion no esta definida cuando k_2 diverge, y en ese regimen la
    calibracion que este modulo realiza carece de sentido.
    """
    if familia == "sn":
        return 1.0, 1.0

    if familia == "st":
        # W ~ Gamma(nu/2, nu/2):  E[W^s] = Gamma(nu/2 + s)/Gamma(nu/2) (nu/2)^{-s}
        if nu <= 2.0:
            raise ValueError(
                f"familia='st' con nu={nu}: se requiere nu > 2 para que la "
                "innovacion tenga varianza finita."
            )
        k1 = float(np.sqrt(nu / 2.0) * np.exp(gammaln((nu - 1.0) / 2.0)
                                              - gammaln(nu / 2.0)))
        k2 = float(nu / (nu - 2.0))
        return k1, k2

    if familia == "sl":
        # W ~ Beta(nu, 1):  E[W^s] = nu / (nu + s)
        if nu <= 1.0:
            raise ValueError(
                f"familia='sl' con nu={nu}: se requiere nu > 1 para que la "
                "innovacion tenga varianza finita."
            )
        return float(nu / (nu - 0.5)), float(nu / (nu - 1.0))

    if familia == "scn":
        # W = escala con probabilidad nu, W = 1 con probabilidad 1 - nu
        if not (0.0 < nu < 1.0):
            raise ValueError(
                f"familia='scn' con nu={nu}: nu es la proporcion de "
                "contaminacion y debe pertenecer a (0, 1)."
            )
        if not (0.0 < escala_contaminacion <= 1.0):
            raise ValueError(
                f"escala_contaminacion={escala_contaminacion}: debe pertenecer "
                "a (0, 1]; valores pequenos producen atipicos mas extremos."
            )
        k1 = float(nu / np.sqrt(escala_contaminacion) + (1.0 - nu))
        k2 = float(nu / escala_contaminacion + (1.0 - nu))
        return k1, k2

    raise ValueError(f"familia='{familia}' invalida. Opciones: {list(_FAMILIAS)}.")


def extraer_factor_escala(
    familia: str,
    nu: float,
    escala_contaminacion: float,
    rng: np.random.Generator,
    tamano: int = 1,
) -> np.ndarray:
    """Extrae `tamano` realizaciones del factor de escala W de la familia."""
    if familia == "sn":
        return np.ones(tamano)
    if familia == "st":
        return rng.gamma(shape=nu / 2.0, scale=2.0 / nu, size=tamano)
    if familia == "sl":
        return rng.beta(nu, 1.0, size=tamano)
    if familia == "scn":
        contaminada = rng.random(tamano) < nu
        return np.where(contaminada, escala_contaminacion, 1.0)
    raise ValueError(f"familia='{familia}' invalida. Opciones: {list(_FAMILIAS)}.")


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenario4(ConfigObservacion):
    """
    Parametros del Escenario 4.

    Hereda de `ConfigObservacion` los parametros del esquema de observacion
    (L, T, burn_in, sigma_obs, R, seed, media_fn, jitter) y agrega los del
    mecanismo generador.

    Operador autorregresivo e innovacion (identicos al Escenario 1)
        gamma     : alcance de la dependencia entre puntos del dominio en psi.
        hs_norm   : valor objetivo de ||Psi||_HS, en (0, 1). Valores altos
                    atenuan la asimetria observable en la curva por agregacion
                    de innovaciones pasadas.
        sigma_eps : escala de la innovacion. Fija la varianza puntual de eps
                    con independencia de la asimetria y de las colas, gracias
                    a la calibracion de Gamma.
        ell       : longitud de correlacion de la innovacion.

    Ley de la innovacion
        familia              : "sn", "st", "sl" o "scn".
        delta_skew           : coeficiente de asimetria en (-1, 1), parametro
                               de forma estandar de la familia skew-normal. El
                               signo determina la direccion de la asimetria; el
                               valor cero recupera el caso simetrico, que
                               reduce el escenario a un FAR(1) con innovacion
                               de la familia sin asimetria; los valores
                               extremos producen la asimetria maxima admisible
                               (coeficiente de asimetria de 0.995 en el caso
                               skew-normal). Es adimensional e independiente de
                               la grilla, de modo que un mismo valor produce la
                               misma forma distribucional con cualquier L y
                               cualquier ell.
        forma_skew_fn        : forma tau -> d(tau) que modula el coeficiente de
                               asimetria a lo largo del dominio, normalizada a
                               maximo unitario en valor absoluto, de modo que
                               delta_l = delta_skew * d(tau_l). Por defecto
                               constante, caso en que la asimetria afecta por
                               igual a todo el dominio; una forma localizada la
                               concentra en una region.
        nu                   : parametro de la mezcla de escala. En "st" y "sl"
                               gobierna el peso de las colas y valores pequenos
                               las hacen mas pesadas; en "scn" es la proporcion
                               de contaminacion, en (0, 1). No se emplea en "sn".
        escala_contaminacion : factor de escala de la componente contaminada,
                               en (0, 1]. Solo se emplea en "scn".
    """

    # Dinamica (identica al Escenario 1)
    gamma: float = 0.3
    hs_norm: float = 0.5
    sigma_eps: float = 1.0
    ell: float = 0.2

    # Ley de la innovacion
    familia: str = "st"
    delta_skew: float = 0.85
    forma_skew_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    nu: float = 5.0
    escala_contaminacion: float = 0.2

    def validar(self) -> None:
        super().validar()
        if not (0.0 < self.hs_norm < 1.0):
            raise ValueError(
                f"hs_norm={self.hs_norm}: debe estar en (0, 1) para garantizar "
                "la existencia de una solucion estacionaria."
            )
        if self.gamma <= 0:
            raise ValueError("gamma debe ser positivo.")
        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")
        if self.sigma_eps <= 0:
            raise ValueError("sigma_eps debe ser positivo.")
        if not (-1.0 < self.delta_skew < 1.0):
            raise ValueError(
                f"delta_skew={self.delta_skew}: el coeficiente de asimetria debe "
                "pertenecer a (-1, 1). Los extremos corresponden al caso "
                "degenerado en que toda la dispersion procede de la componente "
                "asimetrica."
            )
        if self.familia not in _FAMILIAS:
            raise ValueError(
                f"familia='{self.familia}' invalida. Opciones: {list(_FAMILIAS)}."
            )
        # Valida la existencia de los momentos inversos de la familia elegida
        momentos_mezcla_escala(self.familia, self.nu, self.escala_contaminacion)


# ==========================================================================
# CONSTRUCCION DE LA INNOVACION
# ==========================================================================

def _evaluar_forma_skew(
    forma_fn: Optional[Callable[[np.ndarray], np.ndarray]], tau: np.ndarray
) -> np.ndarray:
    """Forma de la asimetria, normalizada a maximo unitario en valor absoluto."""
    if forma_fn is None:
        return np.ones_like(tau)
    d = np.asarray(forma_fn(tau), dtype=float).ravel()
    if d.shape != tau.shape:
        raise ValueError(
            f"forma_skew_fn retorno forma {d.shape}; se esperaba {tau.shape}."
        )
    pico = float(np.max(np.abs(d)))
    if pico <= 0:
        raise ValueError("forma_skew_fn produjo una funcion nula.")
    return d / pico


def construir_componentes_smsn(
    tau: np.ndarray, cfg: ConfigEscenario4
) -> dict:
    """
    Construye los componentes de la innovacion SMSN calibrados a la varianza
    puntual objetivo sigma_eps^2.

    A partir del coeficiente de asimetria delta_l = delta_skew * d(tau_l) se
    despejan la magnitud de la componente asimetrica y la varianza de la
    componente gaussiana,

        Delta_l  = sigma_eps * delta_l / sqrt(den_l),
        Gamma_ll = sigma_eps^2 * (1 - delta_l^2) / den_l,
        den_l    = coef * delta_l^2 + k_2 (1 - delta_l^2),

    con coef = k_2 - 2 k_1^2 / pi, y se completa Gamma = D C D con D el
    reescalado diagonal y C la correlacion exponencial cuadratica. La igualdad
    Var(eps_l) = coef Delta_l^2 + k_2 Gamma_ll = sigma_eps^2 se satisface de
    forma exacta, y Gamma es semidefinida positiva por construccion, de modo
    que ninguna combinacion de parametros admisibles produce una
    especificacion invalida.

    Retorna un diccionario con Delta, Gamma, su factor de Cholesky, la
    covarianza teorica de la innovacion, los momentos (k_1, k_2) y el vector de
    coeficientes de asimetria marginales.
    """
    k1, k2 = momentos_mezcla_escala(cfg.familia, cfg.nu, cfg.escala_contaminacion)
    coef = k2 - 2.0 * (k1 ** 2) / np.pi

    d_forma = _evaluar_forma_skew(cfg.forma_skew_fn, tau)
    delta = cfg.delta_skew * d_forma                       # coeficiente por punto
    den = coef * delta ** 2 + k2 * (1.0 - delta ** 2)

    Delta = cfg.sigma_eps * delta / np.sqrt(den)
    var_gamma = (cfg.sigma_eps ** 2) * (1.0 - delta ** 2) / den

    # Correlacion exponencial cuadratica, reescalada punto a punto
    C = matriz_covarianza_innovacion(tau, 1.0, cfg.ell)
    D = np.sqrt(var_gamma)
    Gamma = (D[:, None] * C) * D[None, :]
    Gamma = 0.5 * (Gamma + Gamma.T)

    chol_Gamma = factor_cholesky(Gamma, cfg.jitter)

    varianza_teorica = coef * np.outer(Delta, Delta) + k2 * Gamma
    Omega_gauss = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)

    return {
        "Delta": Delta,
        "Gamma": Gamma,
        "chol_Gamma": chol_Gamma,
        "Omega": varianza_teorica,
        "Omega_gaussiana": Omega_gauss,
        "k1": k1,
        "k2": k2,
        "varianza_teorica": varianza_teorica,
        "delta_marginal": delta,
        "centrado": k1 * _SQRT_2_PI * Delta,
    }


def generador_innovacion_smsn(
    componentes: dict, cfg: ConfigEscenario4, rng: np.random.Generator
) -> Callable[[], np.ndarray]:
    """
    Retorna una funcion sin argumentos que extrae una realizacion centrada de
    la innovacion SMSN.

    Se entrega como clausura, en paralelo a `generador_innovacion` de
    `sim_comun`, para que la recursion temporal invoque `innovacion()` sin
    arrastrar los componentes y el generador aleatorio en cada llamada.
    """
    Delta = componentes["Delta"]
    chol_Gamma = componentes["chol_Gamma"]
    centrado = componentes["centrado"]
    L = Delta.size
    familia, nu, escala = cfg.familia, cfg.nu, cfg.escala_contaminacion

    def innovacion() -> np.ndarray:
        w = float(extraer_factor_escala(familia, nu, escala, rng, 1)[0])
        u0 = abs(float(rng.standard_normal()))
        u1 = rng.standard_normal(L)
        return (Delta * u0 + chol_Gamma @ u1) / np.sqrt(w) - centrado

    return innovacion


# ==========================================================================
# DINAMICA DEL ESCENARIO
# ==========================================================================

def simular_trayectoria_far_smsn(
    Psi: np.ndarray,
    innovacion: Callable[[], np.ndarray],
    mu: np.ndarray,
    T: int,
    burn_in: int,
) -> np.ndarray:
    """
    Itera la recursion FAR(1) discretizada con una innovacion arbitraria y
    devuelve las T curvas retenidas SIN ruido de medicion, de dimension (T, L).

    La recursion es identica a la del Escenario 1; lo unico que cambia es la
    ley de `innovacion`, que se recibe como clausura. Separar la recursion de
    la ley de la innovacion es lo que permite que ambos escenarios compartan
    exactamente la misma dinamica y difieran solo en el supuesto distribucional.
    """
    L = Psi.shape[0]
    if mu.shape[0] != L:
        raise ValueError(
            f"Dimensiones inconsistentes: Psi es {Psi.shape} y mu es {mu.shape}."
        )

    Y = innovacion()
    for _ in range(burn_in):
        Y = Psi @ Y + innovacion()

    curvas = np.empty((T, L))
    for t in range(T):
        Y = Psi @ Y + innovacion()
        curvas[t] = mu + Y
    return curvas


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_4(cfg: ConfigEscenario4) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario 4.

    El operador y los componentes calibrados de la innovacion se construyen una
    sola vez y se comparten entre replicas, dado que no dependen de la
    realizacion aleatoria. Cada replica recibe una semilla derivada de la
    semilla maestra, de modo que la replica r es identica con independencia de
    cuantas replicas se generen o del orden de ejecucion.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)

    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm)
    componentes = construir_componentes_smsn(tau, cfg)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        innovacion = generador_innovacion_smsn(componentes, cfg, rng)
        curvas_r = simular_trayectoria_far_smsn(
            Psi, innovacion, mu, cfg.T, cfg.burn_in
        )
        curvas[r] = curvas_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={
            "operador": Psi,
            "cov_innovacion": componentes["Omega"],
            "cov_innovacion_gaussiana": componentes["Omega_gaussiana"],
            "delta_skew": componentes["Delta"],
            "cov_gamma": componentes["Gamma"],
            "chol_gamma": componentes["chol_Gamma"],
            "momentos_mezcla": (componentes["k1"], componentes["k2"]),
            "varianza_teorica_innovacion": componentes["varianza_teorica"],
            "delta_marginal": componentes["delta_marginal"],
            "pesos_cuadratura": w_quad,
        },
    )
    salida.diagnostico = resumen_escenario_4(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def _asimetria_curtosis(X: np.ndarray, eje: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Asimetria y curtosis (no exceso) estandarizadas a lo largo de `eje`."""
    Z = X - X.mean(axis=eje, keepdims=True)
    m2 = np.mean(Z ** 2, axis=eje)
    m3 = np.mean(Z ** 3, axis=eje)
    m4 = np.mean(Z ** 4, axis=eje)
    sd = np.sqrt(np.where(m2 > 0, m2, np.nan))
    return m3 / sd ** 3, m4 / sd ** 4


def _muestra_innovaciones(
    salida: SalidaSimulacion, n: int = 20000, seed: int = 987654321
) -> np.ndarray:
    """
    Extrae una muestra independiente de la innovacion a partir de los
    componentes almacenados, con semilla propia.

    La muestra no interviene en la simulacion y solo sirve para verificar
    empiricamente la calibracion de la varianza y la asimetria efectivamente
    inducida, cantidades que la curva no revela de forma directa por la
    atenuacion autorregresiva.
    """
    cfg = salida.config
    componentes = {
        "Delta": salida.internos["delta_skew"],
        "chol_Gamma": salida.internos["chol_gamma"],
        "centrado": salida.internos["momentos_mezcla"][0] * _SQRT_2_PI
                    * salida.internos["delta_skew"],
    }
    rng = np.random.default_rng(seed)
    innovacion = generador_innovacion_smsn(componentes, cfg, rng)
    return np.array([innovacion() for _ in range(n)])


def _exceso_correlacion(salida: SalidaSimulacion) -> float:
    """
    Exceso maximo de correlacion a larga distancia respecto de la referencia
    gaussiana, atribuible al choque asimetrico comun.

    Es una lectura del precio de introducir asimetria mediante una componente
    de rango uno: la varianza puntual queda igualada al Escenario 1, pero la
    correlacion entre puntos alejados del dominio no. Un valor proximo a cero
    indica que la asimetria es localizada y el contraste con el Escenario 1
    aisla limpiamente el efecto distribucional.
    """
    V = salida.internos.get("varianza_teorica_innovacion")
    O = salida.internos.get("cov_innovacion_gaussiana")
    if V is None or O is None:
        return float("nan")
    sdV = np.sqrt(np.diag(V)); sdO = np.sqrt(np.diag(O))
    RV = V / np.outer(sdV, sdV); RO = O / np.outer(sdO, sdO)
    return float(np.max(np.abs(RV - RO)))


def resumen_escenario_4(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con las verificaciones propias del Algoritmo 4.

    Estacionariedad. Se reporta la norma de Hilbert-Schmidt efectiva del
    operador, cuya condicion es identica a la del Escenario 1: la ley de la
    innovacion no altera el requisito de contractividad, siempre que su
    varianza sea finita, lo que la validacion de `nu` garantiza.

    Calibracion de la escala. Se contrasta la varianza puntual teorica de la
    innovacion contra sigma_eps^2 y se verifica empiricamente sobre una muestra
    independiente. Que ambas coincidan es lo que permite atribuir cualquier
    diferencia de desempeno respecto del Escenario 1 a la forma de la
    distribucion y no a un cambio de razon senal-ruido.

    Asimetria y colas. Se reportan la asimetria y la curtosis de la innovacion
    y de las curvas, junto con su razon. La asimetria de la curva no coincide
    con la de la innovacion, pues la recursion agrega innovaciones pasadas, y
    la direccion del efecto depende de la persistencia y de la longitud de
    correlacion segun se detalla en la documentacion del modulo. Una asimetria
    de curva proxima a cero indica que el escenario no discriminara entre
    metodos por mas asimetrica que sea la innovacion, y aconseja revisar la
    combinacion de delta_skew, hs_norm y ell.
    """
    if not isinstance(salida.config, ConfigEscenario4):
        raise TypeError(
            "resumen_escenario_4 requiere una salida generada con "
            f"ConfigEscenario4; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config

    Psi = salida.internos.get("operador")
    w_quad = salida.internos.get("pesos_cuadratura")
    var_teorica = salida.internos.get("varianza_teorica_innovacion")
    for nombre, objeto in (
        ("operador", Psi), ("pesos_cuadratura", w_quad),
        ("varianza_teorica_innovacion", var_teorica),
    ):
        if objeto is None:
            raise KeyError(
                f"La salida no contiene '{nombre}' en `internos`; no puede "
                "completarse el control de calidad del Escenario 4."
            )

    hs = norma_hilbert_schmidt(Psi, w_quad)
    radio = float(np.max(np.abs(np.linalg.eigvals(Psi))))
    k1, k2 = salida.internos["momentos_mezcla"]

    # Calibracion: varianza puntual teorica frente al objetivo
    var_objetivo = float(cfg.sigma_eps ** 2)
    var_diag = np.diag(var_teorica)
    error_calibracion = float(np.max(np.abs(var_diag - var_objetivo)))

    # Verificacion empirica sobre una muestra independiente de innovaciones
    eps = _muestra_innovaciones(salida)
    asim_eps, curt_eps = _asimetria_curtosis(eps, eje=0)
    var_eps_emp = eps.var(axis=0, ddof=1)

    # Momentos de las curvas: atenuados por la agregacion autorregresiva
    asim_curvas = np.empty((salida.curvas.shape[0], salida.curvas.shape[2]))
    curt_curvas = np.empty_like(asim_curvas)
    for r in range(salida.curvas.shape[0]):
        asim_curvas[r], curt_curvas[r] = _asimetria_curtosis(salida.curvas[r], eje=0)

    delta_marginal = salida.internos.get("delta_marginal")

    # Asimetria marginal teorica; disponible en forma cerrada para skew-normal
    if cfg.familia == "sn":
        d = np.asarray(delta_marginal, dtype=float)
        numerador = ((4.0 - np.pi) / 2.0) * (d * _SQRT_2_PI) ** 3
        denominador = (1.0 - 2.0 * d ** 2 / np.pi) ** 1.5
        asim_teorica = float(np.nanmean(numerador / denominador))
    else:
        asim_teorica = float("nan")

    especifico = {
        "familia": cfg.familia,
        "delta_skew": float(cfg.delta_skew),
        "nu": float(cfg.nu) if cfg.familia != "sn" else float("nan"),
        "k1_E_W_menos_media": float(k1),
        "k2_E_W_inverso": float(k2),
        "hs_norm_objetivo": float(cfg.hs_norm),
        "hs_norm_efectiva": float(hs),
        "hs_norm_error_absoluto": abs(float(hs) - float(cfg.hs_norm)),
        "radio_espectral_operador": radio,
        "estacionariedad_garantizada": bool(hs < 1.0),
        "var_innovacion_objetivo": var_objetivo,
        "var_innovacion_teorica_error_max": error_calibracion,
        "var_innovacion_empirica_media": float(var_eps_emp.mean()),
        "delta_marginal_medio": float(np.nanmean(delta_marginal)),
        "correlacion_larga_distancia_extra": _exceso_correlacion(salida),
        "asimetria_innovacion_teorica": asim_teorica,
        "asimetria_innovacion_empirica": float(np.nanmean(asim_eps)),
        "curtosis_innovacion_empirica": float(np.nanmean(curt_eps)),
        "asimetria_curvas_media": float(np.nanmean(asim_curvas)),
        "curtosis_curvas_media": float(np.nanmean(curt_curvas)),
        "razon_asimetria_curva_innovacion": float(
            np.nanmean(asim_curvas) / np.nanmean(asim_eps)
            if abs(np.nanmean(asim_eps)) > 1e-12 else np.nan
        ),
    }
    return {**base, **especifico}
