"""
sim_escenario_6.py
==================
Escenario 6: proceso con covarianza no estacionaria (Algoritmo 6 del anexo).

Modelo generador
----------------
    a_tj = phi_j a_{t-1,j} + e_tj,    e_tj ~ N(0, lambda_j(t) (1 - phi_j^2)),

    X_t(tau) = mu(tau) + sum_{j=1}^{J} a_tj varphi_j(tau),

sobre la misma construccion del Algoritmo 5 ---sistema ortonormal fijo, base de
Fourier truncada en J terminos, curva obtenida por un unico producto
matricial--- con la unica diferencia de que el espectro varia con el periodo:
lambda_j(t) es una trayectoria DETERMINISTA que interpola entre un espectro
inicial {lambda_j^(0)} y uno final {lambda_j^(1)}.

Que somete a prueba
-------------------
Este escenario pertenece al segundo bloque del anexo y, como el Algoritmo 5, no
interviene la ley condicional: esta sigue siendo gaussiana y unimodal. Lo que
interviene es el supuesto de ESTACIONARIEDAD DE SEGUNDO ORDEN sobre el que
descansa el tratamiento de la media y de las autofunciones como cantidades
conocidas, estimadas una unica vez sobre el bloque de entrenamiento.

Cuando el reordenamiento entre {lambda_j^(0)} y {lambda_j^(1)} altera el ORDEN
de las varianzas ---la configuracion del anexo intercambia las dos primeras
componentes--- la base ajustada sobre el entrenamiento deja de estar ordenada
segun la variabilidad del bloque de prueba. La direccion que el truncamiento
retiene como dominante pasa a ser subordinada justo en el tramo que se evalua.

Conviene precisar el alcance del resultado: ese supuesto NO es relajado por el
modelo dinamico, que flexibiliza la forma de la ley condicional y no la
estructura de covarianza que sostiene la representacion. La degradacion
esperada afecta por igual a todo metodo que opere sobre la misma reduccion, de
modo que el escenario no discrimina entre especificaciones dinamicas sino que
acota el alcance de la reduccion.

Los dos mecanismos de interpolacion
-----------------------------------
La forma de lambda_j(t) distingue dos situaciones que el anexo trata como
cualitativamente distintas y que aqui se seleccionan con `modo`:

"quiebre"  Cambio abrupto en un instante t* situado dentro del bloque de
           prueba. Representa un quiebre estructural: el entrenamiento no
           contiene informacion alguna sobre el espectro que regira la
           evaluacion. Es la configuracion del Cuadro de escenarios, con
           t* = 270 sobre una serie de T = 300 y T0 = 240.

"lineal"   Interpolacion lineal a lo largo de toda la serie retenida.
           Representa una deriva gradual: el entrenamiento contiene una parte
           del desplazamiento y una base estimada sobre el promedia espectros
           distintos, de modo que el deterioro es continuo y no localizado.

Adaptacion de la varianza tras el quiebre
-----------------------------------------
La varianza marginal del coeficiente no salta con lambda_j(t): la recursion
implica v_t = phi_j^2 v_{t-1} + lambda_j(t)(1 - phi_j^2), de modo que la
brecha respecto del nuevo nivel se cierra geometricamente a razon phi_j^2. Con
phi_j = 0.7 la razon es 0.49 y el 95 % de la brecha se cierra en unos cinco
periodos. Esto NO es un defecto del generador sino la consecuencia inevitable
de imponer un quiebre sobre un proceso con memoria; el diagnostico lo reporta
como `periodos_adaptacion` y excluye esa ventana al comparar las varianzas
empiricas antes y despues, para no atribuir al generador una discrepancia que
es propia de la dinamica.

Que se reutiliza y que no
-------------------------
Igual que el Algoritmo 5: de `sim_comun.py` solo el esquema de observacion
(grilla, media, semillas, ruido de medicion y control de calidad transversal).
NO hay operador integral, ni covarianza funcional que factorizar, ni norma de
Hilbert-Schmidt que calibrar. La base, el espectro geometrico y la funcion
media se toman de `sim_escenario_5` para que ambos escenarios del segundo
bloque compartan literalmente la misma representacion y la comparacion entre
ellos aisle el efecto de la no estacionariedad.

Uso tipico desde un notebook
----------------------------
    from model_psbp_fd.pipelines import (
        ConfigEscenario6, generar_escenario_6, guardar_escenario
    )

    cfg = ConfigEscenario6()      # valores del anexo, sin argumentos
    salida = generar_escenario_6(cfg)
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

# La representacion es literalmente la del Algoritmo 5 y se importa de alli en
# lugar de duplicarse: si la convencion de la base cambiara, ambos escenarios
# deben cambiar a la vez o dejan de ser comparables. Los dos auxiliares de
# diagnostico se importan por la misma razon.
from .sim_escenario_5 import (
    base_fourier,
    espectro_geometrico,
    frecuencia_maxima_base,
    media_senoidal,
    _ortonormalidad,
    _acf1_por_componente,
)

# Definicion canonica del proyecto; aqui solo interviene en el control de
# calidad, no en la generacion (ver docstring de `sim_escenario_5`).
from ..utils.quadrature import pesos_trapezoidales

__all__ = [
    "ConfigEscenario6",
    "generar_escenario_6",
    "resumen_escenario_6",
    "simular_coeficientes_ar1_no_estacionario",
    "trayectoria_espectro",
    "espectro_intercambiado",
]

_MODOS = ("quiebre", "lineal")


# ==========================================================================
# ESPECTRO FINAL POR DEFECTO
# ==========================================================================

def espectro_intercambiado(lambdas: Sequence[float], i: int = 1, j: int = 2) -> np.ndarray:
    """
    Copia del espectro con las componentes i y j intercambiadas, con indices
    BASE-1 en concordancia con la notacion del anexo.

    Con los valores por defecto reproduce el espectro final del Cuadro de
    escenarios: "espectro inicial lambda_j^(0) = 2^{-(j-1)} y final con las dos
    primeras componentes intercambiadas".

    El intercambio ---y no un reescalamiento cualquiera--- es lo que da su
    sentido al escenario: deja el CONJUNTO de varianzas intacto, de modo que la
    varianza total del proceso no cambia y la degradacion observada no puede
    atribuirse a que el bloque de prueba sea simplemente mas o menos variable
    que el de entrenamiento. Lo unico que cambia es el ORDEN, que es
    exactamente el supuesto que sostiene el truncamiento.
    """
    lam = np.array(lambdas, dtype=float)
    if not (1 <= i <= lam.size and 1 <= j <= lam.size):
        raise ValueError(
            f"Indices i={i}, j={j} fuera de rango para un espectro de "
            f"{lam.size} componentes."
        )
    lam[i - 1], lam[j - 1] = lam[j - 1], lam[i - 1]
    return lam


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenario6(ConfigObservacion):
    """
    Parametros del Escenario 6.

    Hereda de `ConfigObservacion` los parametros del esquema de observacion y
    REDEFINE sus valores por defecto para que coincidan con el Cuadro de
    escenarios del Capitulo 3 (L = 48, T = 300, burn_in = 200, sigma_obs = 0.1,
    R = 50, mu(tau) = sin(2 pi tau)), de modo que `ConfigEscenario6()` sin
    argumentos sea el escenario del anexo.

    Sistema ortonormal
        J : numero de terminos de la base de Fourier. El anexo fija J = 10.

    Espectros
        lambdas_inicial : espectro {lambda_j^(0)}, estrictamente decreciente.
                          Si es None se construye con
                          `espectro_geometrico(J, razon_espectro)`.
        lambdas_final   : espectro {lambda_j^(1)}. Si es None se construye
                          intercambiando las componentes
                          `indices_intercambio` del espectro inicial. NO se
                          exige que sea decreciente: que no lo sea es
                          precisamente el mecanismo del escenario.
        razon_espectro  : razon del espectro geometrico por defecto.
        indices_intercambio : par de indices BASE-1 que se intercambian para
                          construir el espectro final por defecto.

    Dinamica
        phis      : coeficientes autorregresivos por componente, con
                    |phi_j| < 1. Si es None se usa `phi_comun` en todas.
        phi_comun : valor comun de phi_j. El anexo fija 0.7 para todo j; que
                    sea comun es deliberado, porque asi la predictibilidad no
                    varia entre componentes y el escenario aisla el efecto del
                    reordenamiento del espectro, sin mezclarlo con el
                    desalineamiento que ya introduce el Algoritmo 5.

    Trayectoria del espectro
        modo      : "quiebre" (cambio abrupto en t_quiebre) o "lineal"
                    (interpolacion a lo largo de toda la serie retenida).
        t_quiebre : instante del cambio abrupto, indice BASE-1 sobre la SERIE
                    RETENIDA (no sobre el calentamiento). El periodo t_quiebre
                    es el primero que se genera con el espectro final. El anexo
                    fija t* = 270, que con T = 300 y T0 = 240 cae dentro del
                    bloque de prueba. Solo interviene con modo="quiebre".
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

    # --- Espectros inicial y final ---
    lambdas_inicial: Optional[Sequence[float]] = None
    lambdas_final: Optional[Sequence[float]] = None
    razon_espectro: float = 0.5
    indices_intercambio: tuple = (1, 2)

    # --- Dinamica por componente ---
    phis: Optional[Sequence[float]] = None
    phi_comun: float = 0.7

    # --- Trayectoria del espectro ---
    modo: str = "quiebre"
    t_quiebre: int = 270

    def espectro_inicial(self) -> np.ndarray:
        """Espectro {lambda_j^(0)} efectivo (J,), resolviendo el default."""
        if self.lambdas_inicial is None:
            return espectro_geometrico(self.J, self.razon_espectro)
        return np.asarray(self.lambdas_inicial, dtype=float)

    def espectro_final(self) -> np.ndarray:
        """Espectro {lambda_j^(1)} efectivo (J,), resolviendo el default."""
        if self.lambdas_final is None:
            i, j = self.indices_intercambio
            return espectro_intercambiado(self.espectro_inicial(), i, j)
        return np.asarray(self.lambdas_final, dtype=float)

    def coeficientes_ar(self) -> np.ndarray:
        """Coeficientes phi_j efectivos (J,), resolviendo el default."""
        if self.phis is None:
            return np.full(self.J, float(self.phi_comun))
        return np.asarray(self.phis, dtype=float)

    def validar(self) -> None:
        super().validar()

        if self.J < 2:
            raise ValueError(
                "J debe ser al menos 2; con una unica componente no existe "
                "orden del espectro que el escenario pueda alterar."
            )

        k_max = frecuencia_maxima_base(self.J)
        if 2 * k_max >= self.L - 1:
            raise ValueError(
                f"L={self.L} es insuficiente para J={self.J}: la cuadratura "
                f"trapezoidal requiere 2*k_max < L-1, con k_max={k_max}. "
                "Aumente L o reduzca J; en caso contrario la base deja de ser "
                "ortonormal en la metrica discreta."
            )

        lam0 = self.espectro_inicial()
        lam1 = self.espectro_final()
        for nombre, lam in (("lambdas_inicial", lam0), ("lambdas_final", lam1)):
            if lam.shape != (self.J,):
                raise ValueError(
                    f"{nombre} tiene forma {lam.shape}; se esperaba ({self.J},)."
                )
            if np.any(lam <= 0):
                raise ValueError(f"{nombre} debe ser estrictamente positivo.")

        # Solo el espectro inicial debe ser decreciente: el final no lo es por
        # construccion, y esa es la caracteristica que define al escenario.
        if not np.all(np.diff(lam0) < 0):
            raise ValueError(
                "lambdas_inicial debe ser estrictamente decreciente: define el "
                "orden de la base que se estima sobre el entrenamiento."
            )
        if np.allclose(lam0, lam1):
            raise ValueError(
                "lambdas_inicial y lambdas_final coinciden: el proceso resulta "
                "estacionario de segundo orden y el escenario no somete a "
                "prueba nada. Use el Escenario 5 si eso es lo buscado."
            )

        i, j = self.indices_intercambio
        if not (1 <= int(i) <= self.J and 1 <= int(j) <= self.J):
            raise ValueError(
                f"indices_intercambio={self.indices_intercambio} fuera de "
                f"[1, {self.J}]."
            )

        phis = self.coeficientes_ar()
        if phis.shape != (self.J,):
            raise ValueError(
                f"phis tiene forma {phis.shape}; se esperaba ({self.J},)."
            )
        if np.any(np.abs(phis) >= 1.0):
            raise ValueError(
                "phis debe cumplir |phi_j| < 1 para que la varianza marginal "
                "siga a lambda_j(t) en lugar de divergir."
            )

        if self.modo not in _MODOS:
            raise ValueError(
                f"modo='{self.modo}' invalido. Opciones: {list(_MODOS)}."
            )
        if self.modo == "quiebre":
            if not (2 <= self.t_quiebre <= self.T):
                raise ValueError(
                    f"t_quiebre={self.t_quiebre}: debe ser un indice base-1 en "
                    f"[2, T] con T={self.T}, de modo que exista al menos un "
                    "periodo bajo cada espectro."
                )


# ==========================================================================
# TRAYECTORIA DETERMINISTA DEL ESPECTRO
# ==========================================================================

def trayectoria_espectro(
    lambdas_inicial: np.ndarray,
    lambdas_final: np.ndarray,
    T: int,
    modo: str = "quiebre",
    t_quiebre: int = 270,
) -> np.ndarray:
    """
    Matriz (T, J) con el espectro lambda_j(t) vigente en cada periodo RETENIDO.

    El periodo de calentamiento no aparece en esta matriz: se genera siempre
    bajo el espectro inicial, de modo que la serie retenida comience en el
    equilibrio de {lambda_j^(0)} y el desplazamiento posterior sea el unico
    fenomeno no estacionario presente.

    modo="quiebre"
        lambda_j(t) = lambda_j^(0) para t < t_quiebre,
                      lambda_j^(1) para t >= t_quiebre,
        con t_quiebre un indice base-1 sobre la serie retenida.

    modo="lineal"
        Interpolacion lineal entre ambos espectros a lo largo de los T periodos
        retenidos, con lambda_j(1) = lambda_j^(0) y lambda_j(T) = lambda_j^(1).
    """
    lam0 = np.asarray(lambdas_inicial, dtype=float)
    lam1 = np.asarray(lambdas_final, dtype=float)
    if lam0.shape != lam1.shape:
        raise ValueError(
            f"Los espectros tienen formas distintas: {lam0.shape} y {lam1.shape}."
        )
    if modo not in _MODOS:
        raise ValueError(f"modo='{modo}' invalido. Opciones: {list(_MODOS)}.")

    if modo == "quiebre":
        if not (2 <= t_quiebre <= T):
            raise ValueError(
                f"t_quiebre={t_quiebre} fuera de [2, T] con T={T}."
            )
        Lam = np.tile(lam0, (T, 1))
        Lam[t_quiebre - 1:] = lam1            # base-1 -> base-0
        return Lam

    peso = np.linspace(0.0, 1.0, T)[:, None]  # 0 en t=1, 1 en t=T
    return (1.0 - peso) * lam0[None, :] + peso * lam1[None, :]


# ==========================================================================
# DINAMICA DEL ESCENARIO
# ==========================================================================

def simular_coeficientes_ar1_no_estacionario(
    Lambda_t: np.ndarray,
    lambdas_inicial: np.ndarray,
    phis: np.ndarray,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Itera las J recursiones AR(1) escalares con varianza de innovacion variable
    en el tiempo y devuelve la matriz de coeficientes retenidos A (T, J).

    Parametros
    ----------
    Lambda_t : (T, J)
        Espectro vigente en cada periodo retenido, tipicamente producido por
        `trayectoria_espectro`.
    lambdas_inicial : (J,)
        Espectro que rige la inicializacion y el calentamiento.

    Como en el Algoritmo 5, las J recursiones son independientes y el paso
    temporal se vectoriza sobre las componentes; la unica diferencia es que la
    desviacion estandar de la innovacion se recalcula en cada periodo a partir
    de la fila correspondiente de `Lambda_t`.

    La inicializacion es estacionaria bajo el espectro INICIAL,
    a_{0j} ~ N(0, lambda_j^(0)), tal como indica el anexo. El calentamiento se
    ejecuta integramente bajo ese mismo espectro: su proposito aqui no es
    alcanzar el equilibrio ---la inicializacion ya esta en el--- sino mantener
    homogeneo el consumo de numeros aleatorios respecto del resto de los
    escenarios.
    """
    Lambda_t = np.asarray(Lambda_t, dtype=float)
    lam0 = np.asarray(lambdas_inicial, dtype=float)
    phis = np.asarray(phis, dtype=float)
    T, J = Lambda_t.shape
    if lam0.shape != (J,) or phis.shape != (J,):
        raise ValueError(
            f"Dimensiones inconsistentes: Lambda_t es {Lambda_t.shape}, "
            f"lambdas_inicial es {lam0.shape} y phis es {phis.shape}."
        )

    factor = 1.0 - phis ** 2                      # correccion de la varianza
    sd_burn = np.sqrt(lam0 * factor)

    a = np.sqrt(lam0) * rng.standard_normal(J)    # a_0 estacionario en lam0
    for _ in range(burn_in):
        a = phis * a + sd_burn * rng.standard_normal(J)

    sd_t = np.sqrt(Lambda_t * factor[None, :])    # (T, J)
    A = np.empty((T, J))
    for t in range(T):
        a = phis * a + sd_t[t] * rng.standard_normal(J)
        A[t] = a
    return A


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_6(cfg: ConfigEscenario6) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario 6.

    La base evaluada sobre la grilla y la trayectoria del espectro son
    deterministas y se construyen una sola vez. Cada replica recibe una semilla
    derivada de la semilla maestra, de modo que la replica r es identica con
    independencia de cuantas replicas se generen o del orden de ejecucion.

    Las curvas se obtienen mediante el mismo producto matricial del Algoritmo 5,
    X = 1 mu^T + A Phi^T.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)

    Phi = base_fourier(tau, cfg.J)
    lam0 = cfg.espectro_inicial()
    lam1 = cfg.espectro_final()
    phis = cfg.coeficientes_ar()
    Lambda_t = trayectoria_espectro(lam0, lam1, cfg.T, cfg.modo, cfg.t_quiebre)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    coeficientes = np.empty((cfg.R, cfg.T, cfg.J))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        A = simular_coeficientes_ar1_no_estacionario(
            Lambda_t, lam0, phis, cfg.burn_in, rng
        )
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
            "espectro_inicial": lam0,
            "espectro_final": lam1,
            "trayectoria_espectro": Lambda_t,
            "coeficientes_ar": phis,
            "pesos_cuadratura": w_quad,
        },
    )
    salida.diagnostico = resumen_escenario_6(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def _periodos_adaptacion(phis: np.ndarray, tolerancia: float = 0.05) -> int:
    """
    Numero de periodos que la varianza marginal necesita para cerrar el
    (1 - tolerancia) de la brecha tras un cambio abrupto del espectro.

    La recursion implica v_t - lambda(t) = phi^2 (v_{t-1} - lambda(t)), de modo
    que la brecha decae como phi^{2n} y el numero buscado es
    n = log(tolerancia) / log(phi_max^2). Con phi = 0 el ajuste es inmediato.
    """
    phi_max = float(np.max(np.abs(np.asarray(phis, dtype=float))))
    if phi_max <= 0.0:
        return 0
    return int(np.ceil(np.log(tolerancia) / np.log(phi_max ** 2)))


def _estimar_instante_quiebre(coeficientes: np.ndarray, i: int, j: int) -> int:
    """
    Estimador CUSUM del instante de quiebre, en indice BASE-1 sobre la serie
    retenida, a partir del contraste de energia entre las componentes i y j
    (indices base-1).

    Se construye d_t = promedio sobre replicas de (a_{ti}^2 - a_{tj}^2), cuya
    esperanza cambia de signo al intercambiarse las varianzas, y se toma el
    argumento que maximiza la suma acumulada centrada. Es el estimador estandar
    de punto de cambio en media para una serie con un unico salto, y aqui
    cumple un rol de verificacion: si el instante estimado no cae en un entorno
    de t*, el espectro no se aplico en el periodo que la configuracion declara.
    """
    d = np.mean(coeficientes[:, :, i - 1] ** 2 - coeficientes[:, :, j - 1] ** 2,
                axis=0)
    cusum = np.cumsum(d - d.mean())
    return int(np.argmax(cusum)) + 1        # base-0 -> base-1


def resumen_escenario_6(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con las verificaciones propias del Algoritmo 6.

    Representacion. Como en el Algoritmo 5, se verifica que la base sea
    ortonormal en la metrica discreta y que la reproyeccion de las curvas
    devuelva los coeficientes generados. Sin esas dos identidades, el
    reordenamiento del espectro no seria observable sobre las curvas.

    Reordenamiento del espectro. La verificacion central. Se comparan las
    varianzas empiricas de cada componente en dos ventanas ---antes del quiebre
    y despues, excluyendo los `periodos_adaptacion` en que la varianza aun
    converge al nuevo nivel--- y se reportan los ordenes de componentes en cada
    una. Se declara `reordenamiento_detectado` cuando el orden de la ventana
    posterior difiere del de la anterior, que es la condicion que hace
    obsoleta la base estimada sobre el entrenamiento.

    Localizacion del quiebre. Se estima el instante de cambio por CUSUM sobre
    el contraste de energia entre las dos componentes intercambiadas y se
    reporta su desviacion respecto de `t_quiebre`. Con modo="lineal" el
    estimador carece de objetivo puntual y se reporta unicamente como
    referencia descriptiva del punto medio de la deriva.

    Potencia del diagnostico. Las dos verificaciones anteriores se calculan
    sobre la ventana posterior al quiebre, que con la configuracion del anexo
    tiene apenas T - t* = 30 periodos, de los cuales se descartan los de
    adaptacion. Con phi_j = 0.7 el tamano muestral efectivo de esa ventana es
    aun menor, de modo que con UNA replica la varianza empirica de las dos
    primeras componentes no alcanza a separarse y `reordenamiento_detectado`
    puede resultar False pese a que el generador es correcto. La cifra es
    informativa a partir de una decena de replicas; con R = 1 debe leerse
    `espectro_post_error_max` y no el orden. Lo mismo aplica a
    `instante_quiebre_estimado`, que promedia sobre replicas antes de acumular.

    Persistencia. Se reporta la autocorrelacion empirica a rezago uno por
    componente, que debe aproximar el phi_j comun. Una discrepancia indica que
    la correccion (1 - phi_j^2) o la trayectoria del espectro se aplicaron mal;
    conviene notar que la variacion del espectro sesga levemente a la baja esta
    autocorrelacion cuando se calcula sobre la serie completa, razon por la
    cual tambien se reporta calculada solo sobre la ventana previa al quiebre.
    """
    if not isinstance(salida.config, ConfigEscenario6):
        raise TypeError(
            "resumen_escenario_6 requiere una salida generada con "
            f"ConfigEscenario6; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config

    Phi = salida.internos.get("base")
    A = salida.internos.get("coeficientes")
    lam0 = salida.internos.get("espectro_inicial")
    lam1 = salida.internos.get("espectro_final")
    phis = salida.internos.get("coeficientes_ar")
    w_quad = salida.internos.get("pesos_cuadratura")
    for nombre, objeto in (
        ("base", Phi), ("coeficientes", A), ("espectro_inicial", lam0),
        ("espectro_final", lam1), ("coeficientes_ar", phis),
        ("pesos_cuadratura", w_quad),
    ):
        if objeto is None:
            raise KeyError(
                f"La salida no contiene '{nombre}' en `internos`; no puede "
                "completarse el control de calidad del Escenario 6."
            )

    R, T, J = A.shape

    # --- Representacion ---
    error_ortonormalidad = _ortonormalidad(Phi, w_quad)
    centradas = salida.curvas - salida.media[None, None, :]
    A_reproyectado = centradas @ (w_quad[:, None] * Phi)
    error_reproyeccion = float(np.max(np.abs(A_reproyectado - A)))

    # --- Ventanas antes y despues del cambio ---
    n_adaptacion = _periodos_adaptacion(phis)
    corte = cfg.t_quiebre - 1 if cfg.modo == "quiebre" else T // 2   # base-0
    fin_post = min(corte + n_adaptacion, T - 1)
    var_pre = A[:, :corte, :].var(axis=1).mean(axis=0)              # (J,)
    var_post = A[:, fin_post:, :].var(axis=1).mean(axis=0)          # (J,)

    orden_pre = (np.argsort(-var_pre) + 1).tolist()                 # base-1
    orden_post = (np.argsort(-var_post) + 1).tolist()
    orden_objetivo_pre = (np.argsort(-np.asarray(lam0)) + 1).tolist()
    orden_objetivo_post = (np.argsort(-np.asarray(lam1)) + 1).tolist()

    # --- Localizacion del quiebre ---
    i, j = (int(cfg.indices_intercambio[0]), int(cfg.indices_intercambio[1]))
    t_estimado = _estimar_instante_quiebre(A, i, j)

    # --- Persistencia ---
    acf1_total = _acf1_por_componente(A)
    acf1_pre = _acf1_por_componente(A[:, :corte, :])

    especifico = {
        "J": int(J),
        "modo_trayectoria": cfg.modo,
        "t_quiebre_objetivo": int(cfg.t_quiebre),
        "periodos_adaptacion": int(n_adaptacion),
        "ortonormalidad_error_max": error_ortonormalidad,
        "base_ortonormal": bool(error_ortonormalidad < 1e-8),
        "reproyeccion_error_max": error_reproyeccion,
        "espectro_inicial_objetivo": np.asarray(lam0).tolist(),
        "espectro_final_objetivo": np.asarray(lam1).tolist(),
        "var_empirica_pre": var_pre.tolist(),
        "var_empirica_post": var_post.tolist(),
        "espectro_pre_error_max": float(np.max(np.abs(var_pre - np.asarray(lam0)))),
        "espectro_post_error_max": float(np.max(np.abs(var_post - np.asarray(lam1)))),
        "orden_componentes_pre": orden_pre,
        "orden_componentes_post": orden_post,
        "orden_objetivo_pre": orden_objetivo_pre,
        "orden_objetivo_post": orden_objetivo_post,
        "reordenamiento_detectado": bool(orden_pre != orden_post),
        "reordenamiento_esperado": bool(orden_objetivo_pre != orden_objetivo_post),
        "orden_pre_correcto": bool(orden_pre == orden_objetivo_pre),
        "orden_post_correcto": bool(orden_post == orden_objetivo_post),
        "instante_quiebre_estimado": int(t_estimado),
        "instante_quiebre_error": (
            int(abs(t_estimado - cfg.t_quiebre)) if cfg.modo == "quiebre"
            else None
        ),
        "phis_objetivo": np.asarray(phis).tolist(),
        "acf1_por_componente": acf1_total.tolist(),
        "acf1_por_componente_pre": acf1_pre.tolist(),
    }
    return {**base, **especifico}
