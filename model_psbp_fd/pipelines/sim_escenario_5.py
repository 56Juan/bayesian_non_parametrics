"""
sim_escenario_5.py
==================
Escenario 5: proceso con predictibilidad en componente subordinada
(Algoritmo 5 del anexo).

Modelo generador
----------------
    a_tj = phi_j a_{t-1,j} + e_tj,    e_tj ~ N(0, lambda_j (1 - phi_j^2)),

    X_t(tau) = mu(tau) + sum_{j=1}^{J} a_tj varphi_j(tau),

con {varphi_j} un sistema ortonormal FIJO sobre [0, 1] ---la base de Fourier
truncada en J terminos--- y lambda_1 > ... > lambda_J > 0 un espectro
prefijado.

Que somete a prueba
-------------------
Este escenario pertenece al segundo bloque del anexo. A diferencia de los
Algoritmos 1 a 4, no interviene la ley condicional: esta es gaussiana, unimodal
y homogenea, y por lo tanto esta perfectamente dentro de lo que cualquier
metodo lineal puede representar. Lo que interviene es la REDUCCION DE DIMENSION
sobre la que todos los metodos operan.

La covarianza marginal del proceso es exactamente diagonal en {varphi_j}, de
modo que la descomposicion espectral recupera la base correcta y los
autovalores poblacionales son los lambda_j prefijados. Lo que falla es el
criterio con que se fija M: ese criterio ordena las direcciones por varianza
marginal, cantidad que no contiene informacion alguna sobre la dinamica. Al
fijar phi_1 = phi_2 = 0 y phi_3 = 0.9, la unica direccion predecible es la
tercera en varianza, de modo que un truncamiento en M = 2 ---perfectamente
razonable segun el porcentaje de varianza explicada--- descarta la unica senal
que el modelo dinamico podria aprovechar. La perdida ocurre ANTES de que el
modelo dinamico se ajuste y ninguna especificacion de este la remedia.

La parametrizacion Var(e_tj) = lambda_j (1 - phi_j^2) es lo que hace posible
ese desacople: garantiza que la varianza marginal estacionaria de la componente
j sea exactamente lambda_j con independencia de phi_j, de modo que el espectro
queda fijado por construccion y es separable de la dinamica. Sin esa
correccion, subir phi_3 inflaria tambien la varianza de la tercera componente y
el escenario dejaria de aislar el efecto que busca medir.

Que se reutiliza y que no
-------------------------
De `sim_comun.py` se reutiliza UNICAMENTE el esquema de observacion: grilla,
funcion media, semillas por replica, ruido de medicion y control de calidad
transversal. NO se reutiliza la maquinaria de operadores integrales
(`matriz_operador_ar`, `matriz_covarianza_innovacion`, `factor_cholesky`,
`norma_hilbert_schmidt`), porque este algoritmo no tiene operador integral: la
dinamica es un conjunto de J recursiones AR(1) ESCALARES desacopladas y la
innovacion funcional nunca se construye sobre la grilla. La curva aparece una
sola vez, al final, mediante el producto matricial X = 1 mu^T + A Phi^T.

Los pesos de cuadratura si se importan, desde su definicion canonica en
`utils.quadrature`, pero solo para el control de calidad: verificar la
ortonormalidad de la base y reproyectar las curvas sobre ella. No intervienen
en la generacion.

Uso tipico desde un notebook
----------------------------
    from model_psbp_fd.pipelines import (
        ConfigEscenario5, generar_escenario_5, guardar_escenario
    )

    cfg = ConfigEscenario5()      # valores del anexo, sin argumentos
    salida = generar_escenario_5(cfg)
    salida.diagnostico            # control de calidad del generador
    X = salida.observaciones      # (R, T, L) -> insumo del pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    grilla_regular,
    evaluar_media,
    semillas_replicas,
    aplicar_ruido_observacion,
    diagnostico_comun,
)

# Definicion canonica del proyecto. Se importa desde `utils` y no desde
# `sim_comun` para dejar explicito que aqui la cuadratura NO participa de la
# generacion ---la dinamica vive sobre coeficientes--- sino unicamente del
# control de calidad de la base.
from ..utils.quadrature import pesos_trapezoidales

__all__ = [
    "ConfigEscenario5",
    "generar_escenario_5",
    "resumen_escenario_5",
    "simular_coeficientes_ar1",
    "base_fourier",
    "frecuencia_maxima_base",
    "espectro_geometrico",
    "phis_componente_predecible",
    "media_senoidal",
]


# ==========================================================================
# FUNCION MEDIA DEL ESTUDIO
# ==========================================================================

def media_senoidal(tau: np.ndarray) -> np.ndarray:
    """
    Funcion media comun a todos los escenarios del estudio, mu(tau) = sin(2 pi tau).

    Se define aqui porque los Algoritmos 5 y 6 la toman como DEFAULT de su
    configuracion, en linea con el Cuadro de escenarios del Capitulo 3: el
    notebook debe poder instanciar `ConfigEscenario5()` sin argumentos y obtener
    el escenario del anexo.

    Notese que sin(2 pi tau) coincide, salvo la constante de normalizacion, con
    la segunda funcion de la base de Fourier de `base_fourier`. Eso es inocuo
    para la generacion ---la media se suma despues de construir los
    coeficientes--- pero conviene tenerlo presente al inspeccionar curvas sin
    centrar: la forma de la media es exactamente la del primer armonico.
    """
    return np.sin(2.0 * np.pi * tau)


# ==========================================================================
# SISTEMA ORTONORMAL FIJO
# ==========================================================================

def base_fourier(tau: np.ndarray, J: int) -> np.ndarray:
    """
    Evalua la base de Fourier truncada en J terminos sobre la grilla y devuelve
    la matriz Phi de dimension (L, J), con Phi[l, j] = varphi_{j+1}(tau_l).

    Convencion de ordenamiento y normalizacion
    ------------------------------------------
    El sistema es ortonormal respecto del producto interno CONTINUO de
    L^2([0, 1]), es decir int_0^1 varphi_i varphi_j = delta_ij:

        varphi_1(tau)     = 1
        varphi_{2k}(tau)  = sqrt(2) sin(2 pi k tau),   k = 1, 2, ...
        varphi_{2k+1}(tau)= sqrt(2) cos(2 pi k tau),   k = 1, 2, ...

    El orden alterna seno y coseno de la misma frecuencia, de modo que
    truncar en J par retiene el par completo del ultimo armonico salvo su
    coseno. Con J = 10 se retienen la constante y las frecuencias k = 1, ..., 4
    completas, mas el seno de k = 5.

    La normalizacion es la CONTINUA, no la discreta: no se divide por sqrt(L)
    ni se ortonormaliza numericamente sobre la grilla. La razon es que el
    espectro {lambda_j} del anexo son varianzas de los coeficientes respecto de
    esa metrica continua, que es tambien la que emplea la FPCA del proyecto
    (problema generalizado C u = lambda W u, con W la Gram de la base). Adoptar
    la normalizacion discreta escalaria todos los lambda_j por un factor comun
    y el diagnostico del espectro dejaria de ser comparable con el valor
    objetivo.

    Ortonormalidad sobre la grilla
    ------------------------------
    La ortonormalidad continua se traslada a la grilla con error de maquina, no
    solo de forma aproximada: sobre una grilla regular que incluye ambos
    extremos, la cuadratura trapezoidal de `pesos_trapezoidales` equivale a la
    regla trapezoidal periodica de L - 1 nodos, que es exacta para polinomios
    trigonometricos de grado menor que L - 1. Basta entonces con que
    2 * k_max < L - 1, condicion que `validar()` verifica. Si se relaja, la base
    deja de ser ortonormal en la metrica discreta y los coeficientes
    reproyectados dejan de coincidir con los generados; el diagnostico
    `ortonormalidad_error_max` lo detecta.
    """
    if J < 1:
        raise ValueError("J debe ser al menos 1.")
    tau = np.asarray(tau, dtype=float)
    Phi = np.empty((tau.size, J))
    Phi[:, 0] = 1.0
    for j in range(1, J):
        k = (j + 1) // 2                      # frecuencia del armonico
        arg = 2.0 * np.pi * k * tau
        Phi[:, j] = np.sqrt(2.0) * (np.sin(arg) if j % 2 == 1 else np.cos(arg))
    return Phi


def frecuencia_maxima_base(J: int) -> int:
    """Armonico k mas alto que interviene en `base_fourier` con J terminos."""
    return J // 2


# ==========================================================================
# ESPECTRO Y COEFICIENTES AUTORREGRESIVOS POR DEFECTO
# ==========================================================================

def espectro_geometrico(J: int, razon: float = 0.5) -> np.ndarray:
    """
    Espectro geometrico lambda_j = razon^(j - 1), j = 1, ..., J.

    Con `razon = 0.5` reproduce el lambda_j = 2^{-(j-1)} del Cuadro de
    escenarios. El decaimiento geometrico es deliberado: produce un espectro
    estrictamente decreciente en el que las dos primeras componentes concentran
    el 75 % de la varianza total, de modo que un criterio de porcentaje de
    varianza explicada trunca en M = 2 de forma perfectamente defendible ---y
    descarta, justamente, la componente predecible.
    """
    if J < 1:
        raise ValueError("J debe ser al menos 1.")
    if not (0.0 < razon < 1.0):
        raise ValueError(
            f"razon={razon}: debe estar en (0, 1) para que el espectro sea "
            "positivo y estrictamente decreciente."
        )
    return razon ** np.arange(J, dtype=float)


def phis_componente_predecible(
    J: int,
    indice_predecible: int = 3,
    phi_predecible: float = 0.9,
) -> np.ndarray:
    """
    Vector de coeficientes autorregresivos con una unica componente predecible:
    phi_j = 0 salvo phi_{indice_predecible} = phi_predecible.

    `indice_predecible` es un indice BASE-1 sobre la base, de modo que el valor
    3 designa la tercera funcion de `base_fourier`, en concordancia con la
    notacion del anexo. Con los valores por defecto se obtiene
    phi_1 = phi_2 = 0, phi_3 = 0.9, phi_j = 0 para j >= 4, que es exactamente la
    fila del Algoritmo 5 en el Cuadro de escenarios.

    Que las componentes dominantes tengan phi_j = 0 exacto no es un detalle
    cosmetico: las convierte en ruido blanco marginalmente independiente, de
    modo que ningun modelo dinamico ---correctamente especificado o no--- puede
    extraer de ellas prediccion alguna. El escenario queda asi libre de la
    ambiguedad de atribuir el mal desempeno a una dinamica debil en lugar de al
    truncamiento.
    """
    if J < 1:
        raise ValueError("J debe ser al menos 1.")
    if not (1 <= indice_predecible <= J):
        raise ValueError(
            f"indice_predecible={indice_predecible}: debe estar en [1, J] con J={J}."
        )
    phis = np.zeros(J, dtype=float)
    phis[indice_predecible - 1] = float(phi_predecible)
    return phis


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenario5(ConfigObservacion):
    """
    Parametros del Escenario 5.

    Hereda de `ConfigObservacion` los parametros del esquema de observacion y
    REDEFINE sus valores por defecto para que coincidan con el Cuadro de
    escenarios del Capitulo 3 (L = 48, T = 300, burn_in = 200, sigma_obs = 0.1,
    R = 50, mu(tau) = sin(2 pi tau)). El proposito es que
    `ConfigEscenario5()` sin argumentos sea el escenario del anexo y no una
    plantilla que el notebook deba completar: la configuracion del estudio vive
    en el codigo, no replicada en cada notebook.

    Sistema ortonormal
        J : numero de terminos de la base de Fourier. El anexo fija J = 10.
            No es un parametro del mecanismo sino de la representacion: define
            la dimension exacta del espacio en que vive el proceso, de modo que
            el error de representacion de una FPCA con M = J es nulo salvo
            ruido de medicion.

    Espectro
        lambdas       : varianzas marginales estacionarias, en orden
                        estrictamente decreciente. Si es None se construye con
                        `espectro_geometrico(J, razon_espectro)`.
        razon_espectro: razon del espectro geometrico por defecto.

    Dinamica
        phis              : coeficientes autorregresivos por componente, con
                            |phi_j| < 1. Si es None se construye con
                            `phis_componente_predecible`.
        indice_predecible : indice BASE-1 de la unica componente con dinamica.
        phi_predecible    : coeficiente autorregresivo de esa componente. Su
                            magnitud gobierna cuanta senal se pierde al
                            truncar: con 0.9 la componente explica el 81 % de
                            su propia varianza a un paso, de modo que la
                            perdida es inequivoca.
    """

    # --- Esquema de observacion: valores del Cuadro de escenarios ---
    L: int = 48
    T: int = 300
    burn_in: int = 200
    sigma_obs: float = 0.1
    R: int = 50
    media_fn: Optional[Callable[[np.ndarray], np.ndarray]] = media_senoidal

    # --- Sistema ortonormal ---
    J: int = 10

    # --- Espectro ---
    lambdas: Optional[Sequence[float]] = None
    razon_espectro: float = 0.5

    # --- Dinamica por componente ---
    phis: Optional[Sequence[float]] = None
    indice_predecible: int = 3
    phi_predecible: float = 0.9

    def espectro(self) -> np.ndarray:
        """Espectro efectivo (J,), resolviendo el default cuando `lambdas` es None."""
        if self.lambdas is None:
            return espectro_geometrico(self.J, self.razon_espectro)
        return np.asarray(self.lambdas, dtype=float)

    def coeficientes_ar(self) -> np.ndarray:
        """Coeficientes phi_j efectivos (J,), resolviendo el default."""
        if self.phis is None:
            return phis_componente_predecible(
                self.J, self.indice_predecible, self.phi_predecible
            )
        return np.asarray(self.phis, dtype=float)

    def validar(self) -> None:
        super().validar()

        if self.J < 2:
            raise ValueError(
                "J debe ser al menos 2; con una unica componente el escenario "
                "no puede desalinear varianza y predictibilidad."
            )

        # La ortonormalidad de la base sobre la grilla exige resolver el
        # armonico mas alto (ver docstring de `base_fourier`).
        k_max = frecuencia_maxima_base(self.J)
        if 2 * k_max >= self.L - 1:
            raise ValueError(
                f"L={self.L} es insuficiente para J={self.J}: la cuadratura "
                f"trapezoidal requiere 2*k_max < L-1, con k_max={k_max}. "
                "Aumente L o reduzca J; en caso contrario la base deja de ser "
                "ortonormal en la metrica discreta."
            )

        lam = self.espectro()
        if lam.shape != (self.J,):
            raise ValueError(
                f"lambdas tiene forma {lam.shape}; se esperaba ({self.J},)."
            )
        if np.any(lam <= 0):
            raise ValueError("lambdas debe ser estrictamente positivo.")
        if not np.all(np.diff(lam) < 0):
            raise ValueError(
                "lambdas debe ser estrictamente decreciente: el anexo exige "
                "lambda_1 > ... > lambda_J > 0, ya que el escenario se define "
                "por el desalineamiento entre ESE orden y el de la "
                "predictibilidad."
            )

        phis = self.coeficientes_ar()
        if phis.shape != (self.J,):
            raise ValueError(
                f"phis tiene forma {phis.shape}; se esperaba ({self.J},)."
            )
        if np.any(np.abs(phis) >= 1.0):
            raise ValueError(
                "phis debe cumplir |phi_j| < 1 para que cada componente admita "
                "una solucion estacionaria con varianza marginal lambda_j."
            )
        if np.all(phis == 0.0):
            raise ValueError(
                "Todos los phi_j son nulos: el proceso resultante es ruido "
                "blanco funcional y el escenario carece de componente "
                "predecible que el truncamiento pueda descartar."
            )

        if not (1 <= self.indice_predecible <= self.J):
            raise ValueError(
                f"indice_predecible={self.indice_predecible}: debe estar en "
                f"[1, {self.J}]."
            )


# ==========================================================================
# DINAMICA DEL ESCENARIO
# ==========================================================================

def simular_coeficientes_ar1(
    lambdas: np.ndarray,
    phis: np.ndarray,
    T: int,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Itera las J recursiones AR(1) escalares desacopladas y devuelve la matriz
    de coeficientes retenidos A, de dimension (T, J).

    Las J recursiones son independientes entre si, de modo que el paso temporal
    se vectoriza sobre las componentes y no requiere bucle interno ni
    factorizacion alguna: a diferencia de los Algoritmos 1 a 4, aqui no hay
    covarianza funcional que factorizar porque la innovacion es diagonal por
    construccion en la base elegida.

    La inicializacion se toma de la distribucion estacionaria,
    a_{0j} ~ N(0, lambda_j), tal como indica el anexo. Con ello el periodo de
    calentamiento resulta estrictamente redundante ---la cadena ya parte en
    equilibrio--- pero se ejecuta de todos modos para que el consumo de numeros
    aleatorios sea homogeneo con el resto de los escenarios y para que
    `burn_in` conserve su significado si en el futuro se altera la
    inicializacion.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    phis = np.asarray(phis, dtype=float)
    if lambdas.shape != phis.shape:
        raise ValueError(
            f"lambdas es {lambdas.shape} y phis es {phis.shape}: deben "
            "coincidir en la dimension J."
        )
    J = lambdas.size

    # Desviacion estandar de la innovacion que fija la varianza marginal en
    # lambda_j con independencia de phi_j (ec. del anexo).
    sd_innovacion = np.sqrt(lambdas * (1.0 - phis ** 2))

    a = np.sqrt(lambdas) * rng.standard_normal(J)      # a_0 estacionario
    for _ in range(burn_in):
        a = phis * a + sd_innovacion * rng.standard_normal(J)

    A = np.empty((T, J))
    for t in range(T):
        a = phis * a + sd_innovacion * rng.standard_normal(J)
        A[t] = a
    return A


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_5(cfg: ConfigEscenario5) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario 5.

    La base evaluada sobre la grilla y el espectro no dependen de la
    realizacion aleatoria y se construyen una sola vez. Cada replica recibe una
    semilla derivada de la semilla maestra, de modo que la replica r es
    identica con independencia de cuantas replicas se generen o del orden de
    ejecucion.

    Las curvas se obtienen mediante el unico producto matricial del anexo,
    X = 1 mu^T + A Phi^T, con A de dimension (T, J) y Phi de dimension (L, J).
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)

    Phi = base_fourier(tau, cfg.J)
    lambdas = cfg.espectro()
    phis = cfg.coeficientes_ar()

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    coeficientes = np.empty((cfg.R, cfg.T, cfg.J))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        A = simular_coeficientes_ar1(lambdas, phis, cfg.T, cfg.burn_in, rng)
        curvas_r = mu[None, :] + A @ Phi.T
        coeficientes[r] = A
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
            "base": Phi,
            "coeficientes": coeficientes,
            "espectro": lambdas,
            "coeficientes_ar": phis,
            "pesos_cuadratura": w_quad,
        },
    )
    salida.diagnostico = resumen_escenario_5(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def _ortonormalidad(Phi: np.ndarray, w_quad: np.ndarray) -> float:
    """
    Desviacion maxima de Phi^T W Phi respecto de la identidad, con W la matriz
    diagonal de pesos de cuadratura. Mide en una sola cifra si la metrica
    discreta reproduce el producto interno de L^2 sobre la base empleada.
    """
    G = Phi.T @ (w_quad[:, None] * Phi)
    return float(np.max(np.abs(G - np.eye(Phi.shape[1]))))


def _acf1_por_componente(coeficientes: np.ndarray) -> np.ndarray:
    """
    Autocorrelacion empirica a rezago uno de cada componente, promediada sobre
    replicas. Entrada (R, T, J), salida (J,).
    """
    R, T, J = coeficientes.shape
    acf = np.empty((R, J))
    for r in range(R):
        Z = coeficientes[r] - coeficientes[r].mean(axis=0, keepdims=True)
        num = np.sum(Z[:-1] * Z[1:], axis=0)
        den = np.sum(Z * Z, axis=0)
        acf[r] = num / np.where(den > 0, den, np.nan)
    return np.nanmean(acf, axis=0)


def resumen_escenario_5(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con las tres verificaciones que el Algoritmo 5
    necesita para ser el escenario que dice ser.

    Ortonormalidad de la base. Si Phi^T W Phi se aleja de la identidad, los
    coeficientes generados dejan de ser las coordenadas de la curva en la
    metrica de L^2 y todo el resto del diagnostico pierde sentido. Se reporta
    ademas el error maximo de REPROYECCION, esto es, la discrepancia entre los
    coeficientes generados y los recuperados como Phi^T W (X - mu): esa
    identidad solo se cumple si la base es ortonormal en la metrica discreta, y
    es la comprobacion de que la reconstruccion X = 1 mu^T + A Phi^T no
    introdujo errores.

    Espectro empirico. La varianza marginal empirica de cada componente debe
    reproducir el lambda_j prefijado, con independencia de phi_j. Un error
    sistematico creciente en j indica que la correccion (1 - phi_j^2) de la
    varianza de innovacion no se aplico y que, por tanto, varianza y
    predictibilidad han quedado confundidas: exactamente lo que el escenario
    debe evitar. Se reporta tambien el porcentaje de varianza acumulada en las
    dos primeras componentes, que es la cifra que justifica el truncamiento
    equivocado.

    Desalineamiento. La verificacion central: la componente de varianza maxima
    debe ser la primera y la de predictibilidad maxima ---medida por |acf(1)|
    empirica--- debe ser `indice_predecible`, la tercera en la configuracion del
    anexo. Si ambos indices coinciden, el escenario colapsa al Algoritmo 1 y no
    somete a prueba nada.
    """
    if not isinstance(salida.config, ConfigEscenario5):
        raise TypeError(
            "resumen_escenario_5 requiere una salida generada con "
            f"ConfigEscenario5; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config

    Phi = salida.internos.get("base")
    A = salida.internos.get("coeficientes")
    lambdas = salida.internos.get("espectro")
    phis = salida.internos.get("coeficientes_ar")
    w_quad = salida.internos.get("pesos_cuadratura")
    for nombre, objeto in (
        ("base", Phi), ("coeficientes", A), ("espectro", lambdas),
        ("coeficientes_ar", phis), ("pesos_cuadratura", w_quad),
    ):
        if objeto is None:
            raise KeyError(
                f"La salida no contiene '{nombre}' en `internos`; no puede "
                "completarse el control de calidad del Escenario 5."
            )

    # --- Representacion ---
    error_ortonormalidad = _ortonormalidad(Phi, w_quad)
    centradas = salida.curvas - salida.media[None, None, :]
    A_reproyectado = centradas @ (w_quad[:, None] * Phi)      # (R, T, J)
    error_reproyeccion = float(np.max(np.abs(A_reproyectado - A)))

    # --- Espectro empirico ---
    var_empirica = A.var(axis=1).mean(axis=0)                 # (J,)
    error_espectro = np.abs(var_empirica - lambdas)
    total = float(np.sum(lambdas))
    var_acumulada_2 = float(np.sum(lambdas[:2]) / total) if total > 0 else float("nan")

    # --- Predictibilidad ---
    acf1 = _acf1_por_componente(A)
    idx_var_max = int(np.argmax(var_empirica)) + 1            # base-1
    idx_pred_max = int(np.nanargmax(np.abs(acf1))) + 1        # base-1

    especifico = {
        "J": int(cfg.J),
        "ortonormalidad_error_max": error_ortonormalidad,
        "base_ortonormal": bool(error_ortonormalidad < 1e-8),
        "reproyeccion_error_max": error_reproyeccion,
        "espectro_objetivo": lambdas.tolist(),
        "espectro_empirico": var_empirica.tolist(),
        "espectro_error_max": float(np.max(error_espectro)),
        "espectro_error_relativo_max": float(np.max(error_espectro / lambdas)),
        "varianza_acumulada_dos_primeras": var_acumulada_2,
        "phis_objetivo": np.asarray(phis).tolist(),
        "acf1_por_componente": acf1.tolist(),
        "indice_varianza_maxima": idx_var_max,
        "indice_predecibilidad_maxima": idx_pred_max,
        "indice_predecible_objetivo": int(cfg.indice_predecible),
        "componente_predecible_correcta": bool(idx_pred_max == cfg.indice_predecible),
        "desalineacion_confirmada": bool(idx_pred_max != idx_var_max),
    }
    return {**base, **especifico}
