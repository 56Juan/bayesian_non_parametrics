"""
sim_escenario_B.py
==================
Escenario B: proceso autorregresivo funcional con SIGNO CONMUTADO por umbral
sobre el estado rezagado. Escenario de DIAGNOSTICO, no es un Algoritmo del
anexo; se nombra con letra por el mismo motivo que el Escenario A.

Modelo generador
----------------
    X_t(tau) = mu(tau) + Y_t(tau),

    Y_t(tau) = s_t * int_0^1 psi(tau, u) Y_{t-1}(u) du + eps_t(tau),

    P(s_t = +1 | X_{t-1}) = Phi( nitidez * (z_{t-1} - umbral) ),
    P(s_t = -1 | X_{t-1}) = 1 - P(s_t = +1 | X_{t-1}),

    z_{t-1} = <Y_{t-1}, e>_{L^2},

con un UNICO operador Psi ---el mismo del Algoritmo 1--- y una direccion e de
norma unitaria que gobierna la conmutacion. El signo s_t multiplica al operador
completo, de modo que los dos regimenes son ANTISIMETRICOS: Psi y -Psi.

Por que existe este escenario
-----------------------------
El cuello de botella del capitulo no es que falten escenarios que rompan un
supuesto, sino que los que existen rompen supuestos que la clase lineal
tampoco necesita. En el Escenario 1 la media condicional es lineal y el FAR(1)
esta correctamente especificado; en el 2 la media condicional es constante y
ningun metodo la puede batir; en el 4 la dinamica es la del 1. En los tres, el
MISE no discrimina, y no por defecto del modelo propuesto sino porque no hay
nada que ganar en la media condicional. El diagnostico RESET de la corrida 21
dice lo mismo sobre los datos reales.

Este escenario construye deliberadamente el caso contrario: la media
condicional es una funcion FUERTEMENTE NO LINEAL del estado rezagado, y su
mejor aproximacion lineal es casi nula. La razon es de simetria. El mejor
predictor lineal de Y_t dado Y_{t-1} depende del proceso solo a traves de la
covarianza cruzada

    C_1 = E[ Y_t (x) Y_{t-1} ] = Psi E[ s(Y_{t-1}) Y_{t-1} (x) Y_{t-1} ],

y el integrando es PAR en el signo de la conmutacion pero IMPAR en Y: si la ley
de Y fuera exactamente simetrica, C_1 seria exactamente el operador nulo y el
mejor predictor lineal seria la media incondicional. La ley estacionaria no es
exactamente simetrica ---la deriva s(Y)Psi Y es par, y eso desplaza la media---
pero la cancelacion es casi completa cuando la direccion de conmutacion e es
ortogonal a la direccion dominante del proceso. Con el nucleo gaussiano del
Algoritmo 1 y e(tau) proporcional a sin(2 pi tau) se obtiene, en una corrida
larga con los parametros por defecto:

    R^2 del mejor predictor LINEAL   ~ 0.00   (0.002, fuera de muestra)
    R^2 de la media condicional      ~ 0.34

es decir, el mejor predictor lineal no mejora a la media incondicional
mientras un tercio de la varianza es predecible. La autocorrelacion puntual a
rezago uno de la serie generada es ~ 0.05: el proceso parece ruido blanco a
cualquier diagnostico lineal. El FAR(1), el VAR sobre scores y
cualquier metodo de la clase lineal homogenea quedan clavados en la media
incondicional; un metodo capaz de representar una media condicional que cambia
de pendiente con el estado tiene, en cambio, casi un tercio de la varianza
disponible. Esa distancia ---y no un supuesto roto en abstracto--- es lo que
hace informativo al escenario, y `resumen_escenario_B` la mide y la reporta:
`r2_lineal_fuera_de_muestra`, `r2_oraculo_fuera_de_muestra` y su razon.

Relacion con el Escenario 3, y por que no es el mismo escenario
---------------------------------------------------------------
El Algoritmo 3 tambien conmuta el operador segun el estado rezagado. Difiere en
dos puntos, y ambos son la correccion de lo que fallo en la corrida 13:

1. Los regimenes son ANTISIMETRICOS (Psi y -Psi) en vez de ser dos operadores
   con distinta persistencia. La separacion entre las dos medias condicionales
   es 2||Psi Y_{t-1}||, proporcional al estado y no a una constante: los
   origenes con estado grande son los mas bimodales, y no hace falta ningun
   desplazamiento de nivel para separar las modas.

2. Por eso mismo NO hay desplazamientos d_j. El desplazamiento del Algoritmo 3
   es de rango uno sobre la direccion constante y se lleva el 94 % de la
   varianza, de modo que subirlo para separar las modas empuja la regla del
   95 % hacia M = 1 y deja al escenario sin eje de interpretabilidad. Aqui la
   separacion la produce la propia dinamica y el espectro no se degrada.

Y hay una consecuencia adicional que ningun otro escenario tiene: la direccion
de conmutacion e NO es la primera componente principal. Con los parametros por
defecto es esencialmente la SEGUNDA, de modo que con M = 1 el modelo NO OBSERVA
la covariable que gobierna la conmutacion y no puede sino fallar como el FAR,
mientras que con M >= 2 si la observa. El barrido en M pasa asi de medir una
degradacion suave a tener un punto de corte predicho de antemano.

Estacionariedad
---------------
La condicion suficiente es la misma del Algoritmo 1, ||Psi||_HS < 1, porque
|s_t| = 1 y la recursion esta dominada en norma por la del operador sin signo:
||Y_t|| <= ||Psi|| ||Y_{t-1}|| + ||eps_t||. El generador la impone y la
verifica.

Lo que NO vive aqui
-------------------
El esquema de observacion, la cuadratura del operador, la innovacion funcional
gaussiana y el control de calidad transversal provienen de `sim_comun.py`. Este
modulo aporta unicamente la dinamica y sus verificaciones especificas.

Uso tipico desde un notebook
----------------------------
    from model_psbp_fd.pipelines import (
        ConfigEscenarioB, generar_escenario_B, guardar_escenario
    )

    cfg = ConfigEscenarioB(
        L=75, T=400, burn_in=200, R=1, seed=41232, sigma_obs=0.25,
        media_fn=media_senoidal,
        gamma=0.60, hs_norm=0.90, sigma_eps=1.0, ell=0.5,
        nitidez=4.0, umbral=0.0,
    )
    salida = generar_escenario_B(cfg)
    salida.diagnostico["razon_oraculo_lineal"]   # cuanto pierde el mejor lineal
    X = salida.observaciones                     # (R, T, L)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    grilla_regular,
    evaluar_media,
    matriz_operador_ar,
    matriz_covarianza_innovacion,
    factor_cholesky,
    generador_innovacion,
    semillas_replicas,
    aplicar_ruido_observacion,
    diagnostico_comun,
    norma_hilbert_schmidt,
    pesos_trapezoidales,
)

__all__ = [
    "ConfigEscenarioB",
    "generar_escenario_B",
    "resumen_escenario_B",
    "simular_trayectoria_far_signo",
    "direccion_oscilatoria",
    "coeficiente_sarle_mezcla_simetrica",
]


# ==========================================================================
# DIRECCION DE CONMUTACION
# ==========================================================================

def direccion_oscilatoria(tau: np.ndarray) -> np.ndarray:
    """
    Direccion e(tau) proporcional a sin(2 pi tau), normalizada a norma unitaria
    en L^2 con la cuadratura del proyecto.

    Es la eleccion por defecto y no es arbitraria. Cumple tres cosas a la vez:

    - Tiene integral nula sobre [0, 1], de modo que z = <Y, e> no mide el NIVEL
      de la curva sino su inclinacion. La direccion constante ---la que usa el
      Algoritmo 3--- si mide el nivel, y como la deriva conmutada es par en Y,
      conmutar sobre el nivel deja una asimetria grande en la ley estacionaria
      y con ella una correlacion lineal residual que el FAR(1) si aprovecha.
      Medido con los parametros por defecto: conmutar sobre la direccion
      constante deja R^2 lineal ~ 0.13 y conmutar sobre esta lo deja en ~ 0.00.

    - Es casi ortogonal a la primera componente principal del proceso ---que
      con un nucleo gaussiano de alcance moderado es practicamente constante---
      y carga casi toda su masa sobre la segunda. Eso es lo que produce el
      punto de corte del barrido en M descrito en el encabezado del modulo.

    - Es suave, de modo que la proyeccion sobrevive a la representacion en base
      B-spline y al truncamiento FPCA. Una direccion oscilatoria de frecuencia
      alta cargaria sobre componentes que la regla del 95 % descarta, y el
      escenario se volveria no identificable por una razon distinta de la que
      quiere estudiar.
    """
    tau = np.asarray(tau, dtype=float)
    w = pesos_trapezoidales(tau)
    e = np.sin(2.0 * np.pi * tau)
    return e / np.sqrt(float(np.sum(w * e * e)))


def _normalizar_direccion(
    direccion_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    tau: np.ndarray,
) -> np.ndarray:
    """Evalua y normaliza a norma unitaria en L^2 la direccion de conmutacion."""
    if direccion_fn is None:
        return direccion_oscilatoria(tau)
    e = np.asarray(direccion_fn(tau), dtype=float).ravel()
    if e.shape != tau.shape:
        raise ValueError(
            f"direccion_fn retorno forma {e.shape}; se esperaba {tau.shape}."
        )
    w = pesos_trapezoidales(tau)
    norma = float(np.sqrt(np.sum(w * e * e)))
    if norma <= 0:
        raise ValueError("direccion_fn produjo una funcion de norma nula.")
    return e / norma


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioB(ConfigObservacion):
    """
    Parametros del Escenario B.

    Hereda de `ConfigObservacion` el esquema de observacion (L, T, burn_in,
    sigma_obs, R, seed, media_fn, jitter) y agrega la dinamica.

    Operador y innovacion ---identicos en forma a los del Algoritmo 1---
        gamma     : alcance del nucleo gaussiano del operador. El valor por
                    defecto 0.60 duplica el 0.30 del Algoritmo 1, y no es un
                    detalle: cuanto mas ancho es el nucleo, mas se concentra el
                    rango de Psi sobre la direccion practicamente constante y
                    mas ortogonal queda a la direccion de conmutacion, que es
                    la condicion bajo la cual la covarianza cruzada se cancela.
                    Medido en una corrida larga con el resto de los parametros
                    fijos, el R^2 del mejor predictor lineal pasa de 0.031 con
                    gamma = 0.30 a 0.002 con gamma = 0.60, mientras el del
                    oraculo sube de 0.327 a 0.341.
        hs_norm   : norma de Hilbert-Schmidt objetivo de Psi. Debe estar en
                    (0, 1). Fija cuanta senal hay disponible: la media
                    condicional es s_t Psi Y_{t-1}, luego su varianza crece con
                    hs_norm y con ella el R^2 del oraculo. El valor por defecto
                    0.90 es mas alto que el 0.70 del Algoritmo 1 a proposito:
                    el escenario debe producir una brecha inequivoca entre el
                    predictor lineal y el oraculo, no una que haya que discutir
                    contra el ruido Monte Carlo con R = 1. Al compararlo con la
                    corrida 20 hay que declarar que la persistencia difiere.
        sigma_eps : escala de la innovacion funcional.
        ell       : longitud de correlacion de la innovacion. El valor por
                    defecto 0.5 es el de los Algoritmos 1 a 3 del anexo, y con
                    el la regla del 95 % retiene dos componentes.

    Conmutacion
        nitidez      : pendiente del probit, en las unidades de z. Interpola
                       entre un umbral duro (nitidez grande: la media
                       condicional es una funcion determinista y discontinua
                       del estado, y practicamente no quedan origenes con dos
                       modas) y una mezcla genuina (nitidez moderada: en torno
                       al umbral la ley condicional tiene dos modas separadas
                       por 2||Psi Y_{t-1}||). El valor por defecto 4.0 deja
                       aproximadamente un tercio de los origenes ambiguos ---la
                       corrida 13 tenia un 14 %--- conservando una brecha
                       oraculo/lineal cercana a un orden de magnitud. Subirlo
                       aumenta la brecha y elimina los origenes bimodales, que
                       son los unicos con contenido para el eje distribucional:
                       es exactamente el intercambio que el escenario existe
                       para poner sobre la mesa.
        umbral       : punto de corte del probit sobre z. Con la direccion por
                       defecto z tiene media practicamente nula, de modo que
                       umbral = 0 equilibra los dos signos; `resumen` reporta
                       la proporcion efectiva.
        signos       : multiplicadores del operador en cada rama. Por defecto
                       (+1, -1), que es el caso antisimetrico. Se deja
                       parametrizado para poder correr el contraste con una
                       conmutacion de magnitud (p. ej. (1.0, 0.2)), que NO
                       cancela la correlacion lineal y sirve para verificar
                       que la cancelacion viene de la antisimetria.
        direccion_fn : direccion e(tau) sobre la que se proyecta el estado
                       rezagado. Por defecto `direccion_oscilatoria`.

    Diagnostico
        n_dim_diagnostico : numero de componentes principales empiricas sobre
                            las que se ajusta el mejor predictor lineal del
                            control de calidad. Con la grilla completa (L = 75
                            regresores y T = 400 observaciones) el R^2 dentro
                            de muestra estaria inflado por sobreajuste y el
                            escenario pareceria menos favorable de lo que es;
                            con la proyeccion a pocas dimensiones y ademas
                            evaluado fuera de muestra, la cifra es la que el
                            capitulo puede citar.
    """

    # Operador e innovacion
    gamma: float = 0.60
    hs_norm: float = 0.90
    sigma_eps: float = 1.0
    ell: float = 0.5

    # Conmutacion
    nitidez: float = 4.0
    umbral: float = 0.0
    signos: Sequence[float] = (1.0, -1.0)
    direccion_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # Diagnostico
    n_dim_diagnostico: int = 5

    def validar(self) -> None:
        super().validar()

        if self.gamma <= 0:
            raise ValueError("gamma debe ser positivo.")
        if not (0.0 < self.hs_norm < 1.0):
            raise ValueError(
                f"hs_norm={self.hs_norm}: debe estar en (0, 1). Con |s_t| = 1 la "
                "condicion suficiente de estacionariedad es la misma del "
                "Algoritmo 1."
            )
        if self.sigma_eps <= 0:
            raise ValueError("sigma_eps debe ser positivo.")
        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")
        if self.nitidez <= 0:
            raise ValueError(
                "nitidez debe ser positivo. Con nitidez -> 0 el signo se sortea "
                "con probabilidad 1/2 con independencia del estado, la media "
                "condicional es nula y el escenario deja de tener contenido."
            )

        signos = np.asarray(self.signos, dtype=float)
        if signos.size != 2:
            raise ValueError(
                f"signos tiene longitud {signos.size}; el mecanismo probit de "
                "este escenario define exactamente dos ramas."
            )
        if np.max(np.abs(signos)) > 1.0 + 1e-12:
            raise ValueError(
                "Ningun multiplicador de `signos` puede exceder 1 en valor "
                "absoluto: la cota ||Psi_efectivo||_HS <= max|s| ||Psi||_HS es "
                "lo que garantiza la contractividad."
            )
        if np.allclose(signos[0], signos[1]):
            raise ValueError(
                "Los dos multiplicadores coinciden: el proceso se reduce al "
                "Algoritmo 1 con hs_norm reescalada y no hay conmutacion."
            )
        if self.n_dim_diagnostico < 1:
            raise ValueError("n_dim_diagnostico debe ser al menos 1.")


# ==========================================================================
# DINAMICA
# ==========================================================================

def simular_trayectoria_far_signo(
    Psi: np.ndarray,
    chol_K: np.ndarray,
    mu: np.ndarray,
    cfg: ConfigEscenarioB,
    direccion: np.ndarray,
    w_quad: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Itera la recursion y devuelve las T curvas retenidas SIN ruido de medicion.

    Retorna
    -------
    curvas        : (T, L) las curvas X_t = mu + Y_t.
    signos        : (T,)   el multiplicador s_t efectivamente sorteado.
    proyecciones  : (T,)   z_{t-1}, el estado rezagado que gobierna el sorteo
                           del signo del instante t. Es el valor sobre el que
                           se evalua el probit, no el estado contemporaneo.
    prob_positivo : (T,)   P(s_t = +1 | X_{t-1}) = Phi(nitidez (z_{t-1} - c)).
    medias_cond   : (T, L) la media condicional VERDADERA del instante t,
                           E[X_t | X_{t-1}] = mu + (p_t s_+ + (1-p_t) s_-) Psi Y_{t-1}.
                           Es el oraculo: la mejor prediccion puntual posible,
                           y la referencia contra la cual se mide cuanto pierde
                           el mejor predictor lineal. Inobservable por
                           construccion; viaja en `internos` y nunca alimenta
                           la estimacion.

    El estado latente es Y_t = X_t - mu: la proyeccion se calcula sobre la curva
    centrada respecto de la media global, de modo que el umbral tiene una
    lectura unica a lo largo de toda la serie.
    """
    L = Psi.shape[0]
    innovacion = generador_innovacion(chol_K, rng)
    s_mas, s_menos = float(cfg.signos[0]), float(cfg.signos[1])

    def _sortear(y_prev: np.ndarray) -> tuple[float, float, float]:
        z = float(np.sum(w_quad * direccion * y_prev))
        p = float(norm.cdf(cfg.nitidez * (z - cfg.umbral)))
        s = s_mas if rng.random() < p else s_menos
        return s, z, p

    # Calentamiento. La trayectoria se inicializa con una innovacion y se
    # descartan los primeros burn_in periodos, de modo que lo retenido provenga
    # de la distribucion estacionaria.
    Y = innovacion()
    for _ in range(cfg.burn_in):
        s, _, _ = _sortear(Y)
        Y = s * (Psi @ Y) + innovacion()

    curvas = np.empty((cfg.T, L))
    signos = np.empty(cfg.T)
    proyecciones = np.empty(cfg.T)
    prob_positivo = np.empty(cfg.T)
    medias_cond = np.empty((cfg.T, L))

    for t in range(cfg.T):
        s, z, p = _sortear(Y)
        arrastre = Psi @ Y                       # Psi Y_{t-1}, comun a las dos ramas
        medias_cond[t] = mu + (p * s_mas + (1.0 - p) * s_menos) * arrastre
        Y = s * arrastre + innovacion()
        signos[t] = s
        proyecciones[t] = z
        prob_positivo[t] = p
        curvas[t] = mu + Y

    return curvas, signos, proyecciones, prob_positivo, medias_cond


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_B(cfg: ConfigEscenarioB) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario B.

    El operador y la factorizacion de la covarianza de innovacion se construyen
    una sola vez y se comparten entre replicas: no dependen de la realizacion
    aleatoria. Cada replica recibe una semilla derivada de la maestra mediante
    SeedSequence, de modo que la replica r es identica con independencia de
    cuantas se generen o del orden de ejecucion.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)
    direccion = _normalizar_direccion(cfg.direccion_fn, tau)

    Psi = matriz_operador_ar(tau, cfg.gamma, cfg.hs_norm)
    K = matriz_covarianza_innovacion(tau, cfg.sigma_eps, cfg.ell)
    chol_K = factor_cholesky(K, cfg.jitter)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    signos = np.empty((cfg.R, cfg.T))
    proyecciones = np.empty((cfg.R, cfg.T))
    prob_positivo = np.empty((cfg.R, cfg.T))
    medias_cond = np.empty((cfg.R, cfg.T, cfg.L))

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        c_r, s_r, z_r, p_r, m_r = simular_trayectoria_far_signo(
            Psi, chol_K, mu, cfg, direccion, w_quad, rng
        )
        curvas[r] = c_r
        signos[r] = s_r
        proyecciones[r] = z_r
        prob_positivo[r] = p_r
        medias_cond[r] = m_r
        observaciones[r] = aplicar_ruido_observacion(c_r, cfg.sigma_obs, rng)

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={
            "operador": Psi,
            "cov_innovacion": K,
            "direccion_estado": direccion,
            "signos": signos,
            "proyeccion_estado": proyecciones,
            "prob_signo_positivo": prob_positivo,
            "media_condicional": medias_cond,
            "pesos_cuadratura": w_quad,
        },
    )
    salida.diagnostico = resumen_escenario_B(salida)
    return salida


# ==========================================================================
# BIMODALIDAD DE LA LEY CONDICIONAL
# ==========================================================================

def coeficiente_sarle_mezcla_simetrica(
    p: np.ndarray, a: np.ndarray, s: float
) -> np.ndarray:
    """
    Coeficiente de bimodalidad de Sarle b = (g1^2 + 1) / g2 de la mezcla

        p N(+a, s^2) + (1 - p) N(-a, s^2),

    calculado de forma EXACTA a partir de sus momentos centrales y no por
    simulacion. Referencias: 1/3 para la gaussiana, 5/9 para la uniforme;
    valores por encima de 5/9 se interpretan como evidencia de bimodalidad.

    Es la ley condicional del escenario proyectada sobre una direccion: `a` es
    el coeficiente de Psi Y_{t-1} sobre esa direccion, `s` la desviacion de la
    innovacion sobre la misma, y `p` la probabilidad del signo positivo. Se
    calcula sobre el GENERADOR, de modo que da el valor que la predictiva del
    modelo deberia alcanzar si estuviera bien ajustada: es la referencia
    *oracle* que a la corrida 13 le falto y sin la cual no se puede decidir si
    un coeficiente bajo en la predictiva es un hallazgo sobre el modelo o un
    defecto del generador.

    Con p = 1/2 y separacion 2a >> s el coeficiente tiende a 1, su cota
    superior; con a -> 0 tiende a 1/3, el valor gaussiano.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    s2 = float(s) ** 2

    # Momentos centrales de la mezcla, con d1 = 2(1-p)a y d2 = -2pa las
    # desviaciones de cada componente respecto de la media de la mezcla.
    q = p * (1.0 - p)
    mu2 = s2 + 4.0 * q * a ** 2
    mu3 = 8.0 * (a ** 3) * q * (1.0 - 2.0 * p)
    mu4 = (
        16.0 * (a ** 4) * (p * (1.0 - p) ** 4 + (1.0 - p) * p ** 4)
        + 24.0 * s2 * q * a ** 2
        + 3.0 * s2 ** 2
    )
    mu2 = np.where(mu2 > 0, mu2, np.nan)
    g1 = mu3 / mu2 ** 1.5
    g2 = mu4 / mu2 ** 2
    return (g1 ** 2 + 1.0) / g2


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def _fpca_empirica(
    Y: np.ndarray, w: np.ndarray, n_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Autofunciones y varianzas del problema generalizado C u = lambda W u sobre
    la muestra centrada Y (T, L), con la MISMA cuadratura del resto del
    proyecto. Se resuelve simetrizando con W^{1/2}, de modo que las
    autofunciones salen ortonormales en L^2 y no en la metrica euclidea.

    Es una FPCA de diagnostico interno del generador, no la del pipeline: la
    del pipeline se ajusta solo con el bloque de entrenamiento y sobre la serie
    representada en base. Aqui se usa la muestra completa a proposito, porque
    lo que se quiere describir es el proceso y no el ajuste.
    """
    n_dim = int(min(n_dim, Y.shape[1]))
    raiz = np.sqrt(w)
    Cov = Y.T @ Y / Y.shape[0]
    Sim = (raiz[:, None] * Cov) * raiz[None, :]
    lam, V = np.linalg.eigh(Sim)
    orden = np.argsort(lam)[::-1][:n_dim]
    U = V[:, orden] / raiz[:, None]
    return np.maximum(lam[orden], 0.0), U


def resumen_escenario_B(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con las verificaciones propias del escenario.
    Tres bloques, y el segundo es la razon de ser del escenario.

    Contractividad y conmutacion. Norma de Hilbert-Schmidt efectiva y radio
    espectral del operador; proporcion de instantes en cada rama, duracion
    media de las rachas y dispersion del estado rezagado z. `nitidez_en_sd_z`
    traduce la nitidez a unidades interpretables: es el cambio del argumento
    del probit por desviacion estandar del estado, y es la cifra que hay que
    mirar para saber si la conmutacion es dura o suave, no `nitidez` a secas.
    `fraccion_origenes_ambiguos` cuenta los instantes con p en [0.25, 0.75],
    que son los unicos en que la ley condicional tiene dos modas visibles.

    Cuanto pierde el mejor predictor lineal. Se proyecta el proceso sobre sus
    primeras `n_dim_diagnostico` componentes principales empiricas, se ajusta
    por minimos cuadrados el mejor predictor lineal del score contemporaneo
    dado el rezagado SOBRE LA PRIMERA MITAD de la serie y se evalua en la
    SEGUNDA, y se compara con el R^2 que alcanza la media condicional
    verdadera en el mismo tramo. La particion evita que el R^2 del lineal
    quede inflado por sobreajuste, que es justamente el sesgo que haria
    parecer favorable al escenario sin serlo. `razon_oraculo_lineal` es el
    cociente entre ambos: es la cifra que resume el escenario, y por debajo de
    ~3 el escenario no cumple su proposito.

    Bimodalidad de la ley condicional. Coeficiente de Sarle EXACTO de la ley
    condicional verdadera proyectada sobre la primera componente principal,
    promediado sobre los origenes ambiguos y sobre los deterministas por
    separado. Es la referencia *oracle* contra la cual el notebook `_04`
    contrasta el mismo coeficiente calculado sobre la predictiva del modelo:
    sin ella, un coeficiente bajo en la predictiva no distingue un fallo del
    modelo de un generador sin contenido ---que es exactamente donde quedo
    estancada la corrida 13.
    """
    if not isinstance(salida.config, ConfigEscenarioB):
        raise TypeError(
            "resumen_escenario_B requiere una salida generada con "
            f"ConfigEscenarioB; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config

    requeridos = (
        "operador", "cov_innovacion", "direccion_estado", "signos",
        "proyeccion_estado", "prob_signo_positivo", "media_condicional",
        "pesos_cuadratura",
    )
    for nombre in requeridos:
        if salida.internos.get(nombre) is None:
            raise KeyError(
                f"La salida no contiene '{nombre}' en `internos`; no puede "
                "completarse el control de calidad del Escenario B."
            )

    Psi = salida.internos["operador"]
    K = salida.internos["cov_innovacion"]
    w = salida.internos["pesos_cuadratura"]
    e_dir = salida.internos["direccion_estado"]
    signos = salida.internos["signos"]
    z = salida.internos["proyeccion_estado"]
    p_pos = salida.internos["prob_signo_positivo"]
    medias = salida.internos["media_condicional"]

    R, T, L = salida.curvas.shape
    s_mas = float(cfg.signos[0])

    # ── Operador ────────────────────────────────────────────────────────────
    hs = norma_hilbert_schmidt(Psi, w)
    radio = float(np.max(np.abs(np.linalg.eigvals(Psi))))

    # ── Conmutacion ─────────────────────────────────────────────────────────
    prop_positivo = float(np.mean(np.isclose(signos, s_mas)))
    transiciones = float(np.mean([int(np.sum(np.diff(signos[r]) != 0)) for r in range(R)]))
    duracion_racha = float(T / (transiciones + 1.0))
    sd_z = float(np.std(z))
    ambiguo = (p_pos > 0.25) & (p_pos < 0.75)
    frac_ambiguos = float(np.mean(ambiguo))

    # ── Cuanto pierde el mejor predictor lineal ─────────────────────────────
    # Se trabaja replica a replica y se promedia, porque el mejor predictor
    # lineal es una propiedad del proceso y no del conjunto de replicas.
    r2_lin, r2_orc, r2_lin_in = [], [], []
    for r in range(R):
        Y = salida.curvas[r] - salida.curvas[r].mean(axis=0, keepdims=True)
        Mc = medias[r] - medias[r].mean(axis=0, keepdims=True)
        _, U = _fpca_empirica(Y, w, cfg.n_dim_diagnostico)
        proy = w[:, None] * U                        # <f, u_k> = f @ (w * u_k)
        S = Y @ proy                                 # (T, n_dim) scores
        Sm = Mc @ proy                               # oraculo en el mismo espacio
        X0, X1, Mo = S[:-1], S[1:], Sm[1:]
        n = X1.shape[0]
        corte = n // 2
        D = np.column_stack([np.ones(n), X0])
        coef, *_ = np.linalg.lstsq(D[:corte], X1[:corte], rcond=None)
        sce = float(np.sum((X1[corte:] - D[corte:] @ coef) ** 2))
        sct = float(np.sum((X1[corte:] - X1[:corte].mean(axis=0)) ** 2))
        r2_lin.append(1.0 - sce / max(sct, 1e-300))
        sco = float(np.sum((X1[corte:] - Mo[corte:]) ** 2))
        r2_orc.append(1.0 - sco / max(sct, 1e-300))
        coef_in, *_ = np.linalg.lstsq(D, X1, rcond=None)
        r2_lin_in.append(
            1.0 - float(np.sum((X1 - D @ coef_in) ** 2))
            / max(float(np.sum((X1 - X1.mean(axis=0)) ** 2)), 1e-300)
        )
    r2_lineal = float(np.mean(r2_lin))
    r2_oraculo = float(np.mean(r2_orc))

    # ── Bimodalidad de la ley condicional verdadera ─────────────────────────
    # Proyectada sobre la primera componente principal de la primera replica:
    # a_t es el coeficiente de Psi Y_{t-1} y s la desviacion de la innovacion,
    # ambos sobre esa misma direccion.
    Y0 = salida.curvas[0] - salida.curvas[0].mean(axis=0, keepdims=True)
    _, U0 = _fpca_empirica(Y0, w, 1)
    u1 = U0[:, 0]
    arrastre = (Y0[:-1] @ Psi.T) @ (w * u1)          # <Psi Y_{t-1}, u1>
    sd_innov_u1 = float(np.sqrt(max(float(u1 @ ((w[:, None] * K) * w[None, :]) @ u1), 0.0)))
    p1 = p_pos[0][1:]
    sarle = coeficiente_sarle_mezcla_simetrica(p1, arrastre, sd_innov_u1)
    amb1 = ambiguo[0][1:]
    sarle_amb = float(np.nanmean(sarle[amb1])) if amb1.any() else float("nan")
    sarle_det = float(np.nanmean(sarle[~amb1])) if (~amb1).any() else float("nan")

    # Un origen tiene dos modas visibles sobre u1 solo si ademas de ser ambiguo
    # su estado es grande: la separacion entre modas es 2|a_t| y crece con
    # ||Y_{t-1}||, de modo que la bimodalidad no es una propiedad del escenario
    # sino de una submuestra identificable de sus origenes. Es la cifra que
    # dice cuantos origenes del bloque de prueba pueden sostener el argumento
    # distribucional del capitulo.
    bimodal = amb1 & (np.abs(arrastre) > sd_innov_u1)

    # Separacion entre las dos medias condicionales, en unidades de la
    # desviacion L^2 de la innovacion: es la distancia entre modas.
    sep = np.sqrt(np.sum((abs(cfg.signos[0] - cfg.signos[1]) * (Y0[:-1] @ Psi.T)) ** 2 * w, axis=1))
    sd_innov_L2 = float(np.sqrt(max(float(np.sum(w * np.diag(K))), 0.0)))

    especifico = {
        # Operador
        "hs_norm_objetivo": float(cfg.hs_norm),
        "hs_norm_efectiva": float(hs),
        "hs_norm_error_absoluto": float(abs(hs - cfg.hs_norm)),
        "radio_espectral": radio,
        "contractividad": bool(hs < 1.0),
        "estacionariedad_garantizada": bool(hs < 1.0),
        # Conmutacion
        "signos": [float(s) for s in cfg.signos],
        "antisimetrico": bool(np.isclose(cfg.signos[0], -cfg.signos[1])),
        "proporcion_signo_positivo": prop_positivo,
        "n_transiciones_media": transiciones,
        "duracion_media_racha": duracion_racha,
        "sd_proyeccion_estado": sd_z,
        "nitidez": float(cfg.nitidez),
        "nitidez_en_sd_z": float(cfg.nitidez * sd_z),
        "fraccion_origenes_ambiguos": frac_ambiguos,
        "n_origenes_ambiguos": int(np.sum(ambiguo) / R),
        # Lo que el escenario existe para medir
        "n_dim_diagnostico": int(cfg.n_dim_diagnostico),
        "r2_lineal_fuera_de_muestra": r2_lineal,
        "r2_lineal_dentro_de_muestra": float(np.mean(r2_lin_in)),
        "r2_oraculo_fuera_de_muestra": r2_oraculo,
        "brecha_oraculo_lineal": float(r2_oraculo - r2_lineal),
        # None ---y no inf--- cuando el lineal no explica nada: la razon no esta
        # definida, e `inf` se serializa como `Infinity`, que no es JSON valido
        # y rompe a cualquier lector estricto. La cifra citable es la brecha.
        "razon_oraculo_lineal": (
            float(r2_oraculo / r2_lineal) if r2_lineal > 1e-6 else None
        ),
        # Forma de la ley condicional
        "separacion_modas_L2_media": float(np.mean(sep)),
        "sd_innovacion_L2": sd_innov_L2,
        "separacion_modas_en_sd_innovacion": float(np.mean(sep) / max(sd_innov_L2, 1e-300)),
        "razon_arrastre_innovacion_u1": float(np.mean(np.abs(arrastre)) / max(sd_innov_u1, 1e-300)),
        "fraccion_origenes_bimodales": float(np.mean(bimodal)),
        "sarle_oraculo_bimodales": (
            float(np.nanmean(sarle[bimodal])) if bimodal.any() else float("nan")
        ),
        "sarle_oraculo_ambiguos": sarle_amb,
        "sarle_oraculo_deterministas": sarle_det,
        "sarle_referencia_uniforme": 5.0 / 9.0,
        "sarle_referencia_gaussiana": 1.0 / 3.0,
    }
    return {**base, **especifico}
