"""
sim_escenario_3.py
==================
Escenario 3: proceso autorregresivo funcional con cambio de regimen
(Algoritmo 3 del anexo).

Modelo generador
----------------
    X_t(tau) = mu(tau) + d_{S_t}(tau)
               + int_0^1 psi_{S_t}(tau, s) (X_{t-1}(s) - mu(s)) ds
               + eps_t(tau),

donde S_t in {1, ..., J} indexa el regimen vigente en el instante t, cada
regimen posee su propio operador autorregresivo Psi_j (calibrado a su norma de
Hilbert-Schmidt objetivo), su propio desplazamiento de nivel d_j y su propia
escala de innovacion.

Este escenario ataca el supuesto de homogeneidad temporal de los modelos FAR,
FMA y FARMA: la relacion entre el presente y el pasado deja de estar gobernada
por un operador fijo y pasa a depender del regimen vigente. Corresponde a la
columna derecha del cuadro de limitaciones estructurales, y ninguna eleccion de
los nucleos de la clase lineal homogenea puede representarlo.

Mecanismos de cambio de regimen
-------------------------------
El modulo implementa tres mecanismos, seleccionables mediante `mecanismo`, que
se distinguen por la informacion que gobierna la transicion:

"probit"  El regimen depende del ESTADO REZAGADO del proceso a traves de un
          probit ordenado sobre la proyeccion escalar

              z_{t-1} = <X_{t-1} - mu, e>_{L^2},

          con e una direccion de norma unitaria en L^2:

              P(S_t <= j | X_{t-1}) = Phi( nitidez * (umbral_j - z_{t-1}) ).

          El parametro `nitidez` interpola entre dos regimenes limite: cuando
          crece, la transicion se aproxima a un umbral duro y la media
          condicional se vuelve una funcion no lineal determinista del estado;
          cuando es moderada, la ley condicional dada X_{t-1} es una mezcla
          genuina de componentes con pesos que dependen del estado. Este es el
          mecanismo que reproduce la estructura que el modelo propuesto busca
          estimar, de modo que constituye el escenario informativo del estudio.

"markov"  El regimen sigue una cadena de Markov oculta con matriz de
          transicion fija, independiente del estado del proceso. La ley
          condicional dada X_{t-1} es una mezcla, pero sus pesos NO dependen
          del estado rezagado. Sirve como contraste: acota cuanto del
          desempeno del modelo propuesto proviene de la mezcla en si y cuanto
          de la dependencia de los pesos respecto del estado. Con filas
          identicas en la matriz de transicion se obtiene el caso de mezcla
          independiente del pasado.

"quiebre" El regimen cambia en instantes deterministas fijados como fracciones
          de la serie retenida. Es un cambio estructural y NO produce un
          proceso estacionario: su proposito es evaluar la robustez del
          esquema de retencion temporal cuando el quiebre cae dentro del
          bloque de entrenamiento o del bloque de prueba.

Estacionariedad
---------------
Para los mecanismos "probit" y "markov", una condicion suficiente de
ergodicidad geometrica es que todos los regimenes sean contractivos, esto es
max_j ||Psi_j||_HS < 1, condicion que el generador impone y verifica. El
mecanismo "quiebre" es no estacionario por construccion y el diagnostico lo
reporta de forma explicita.

El esquema de observacion, la cuadratura del operador, la innovacion funcional
y el control de calidad transversal provienen de `sim_comun.py`; este modulo
aporta unicamente la dinamica del algoritmo y sus verificaciones especificas.

Uso tipico desde un notebook
----------------------------
    from model_psbp_fd.pipelines import (
        ConfigEscenario3, generar_escenario_3, guardar_escenario
    )

    cfg = ConfigEscenario3(
        L=48, T=200, burn_in=200, R=50, seed=20260719,
        mecanismo="probit",
        gammas=(0.30, 0.30), hs_norms=(0.30, 0.85),
        sigmas_eps=(1.0, 1.0), desplazamientos=(-1.5, 1.5),
        umbrales=(0.0,), nitidez=2.0, ell=0.2, sigma_obs=0.5,
    )
    salida = generar_escenario_3(cfg)
    salida.diagnostico            # control de calidad del generador
    X = salida.observaciones      # (R, T, L) -> insumo del pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    "ConfigEscenario3",
    "generar_escenario_3",
    "resumen_escenario_3",
    "simular_trayectoria_far_regimen",
    "direccion_constante",
]

_MECANISMOS = ("probit", "markov", "quiebre")


# ==========================================================================
# DIRECCION DE PROYECCION DEL ESTADO
# ==========================================================================

def direccion_constante(tau: np.ndarray) -> np.ndarray:
    """
    Direccion e(tau) constante, normalizada a norma unitaria en L^2.

    Con esta eleccion la proyeccion z = <X - mu, e> es proporcional al nivel
    medio de la curva centrada, de modo que el regimen responde a si la curva
    completa se situa por encima o por debajo de su media. Es la eleccion por
    defecto por ser la mas interpretable; cualquier otra direccion permite que
    el regimen responda a un rasgo localizado del dominio.
    """
    tau = np.asarray(tau, dtype=float)
    w = pesos_trapezoidales(tau)
    e = np.ones_like(tau)
    return e / np.sqrt(float(np.sum(w * e * e)))


def _normalizar_direccion(
    direccion_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    tau: np.ndarray,
) -> np.ndarray:
    """Evalua y normaliza a norma unitaria en L^2 la direccion de proyeccion."""
    if direccion_fn is None:
        return direccion_constante(tau)
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


def _evaluar_forma(
    forma_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    tau: np.ndarray,
) -> np.ndarray:
    """
    Evalua la forma del desplazamiento de nivel, normalizada a maximo unitario
    en valor absoluto, de modo que el escalar de cada regimen fije la magnitud.
    """
    if forma_fn is None:
        return np.ones_like(tau)
    f = np.asarray(forma_fn(tau), dtype=float).ravel()
    if f.shape != tau.shape:
        raise ValueError(
            f"forma_desplazamiento_fn retorno forma {f.shape}; se esperaba {tau.shape}."
        )
    pico = float(np.max(np.abs(f)))
    if pico <= 0:
        raise ValueError("forma_desplazamiento_fn produjo una funcion nula.")
    return f / pico


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenario3(ConfigObservacion):
    """
    Parametros del Escenario 3.

    Hereda de `ConfigObservacion` los parametros del esquema de observacion
    (L, T, burn_in, sigma_obs, R, seed, media_fn, jitter) y agrega los del
    mecanismo generador. El numero de regimenes J queda determinado por la
    longitud de las tuplas por regimen, que deben ser todas de igual longitud.

    Operadores por regimen
        gammas     : alcance del nucleo psi_j en cada regimen.
        hs_norms   : valor objetivo de ||Psi_j||_HS en cada regimen. Todos
                     deben pertenecer a (0, 1); el contraste entre un regimen
                     poco persistente y otro muy persistente es el que produce
                     dinamicas distinguibles.
        sigmas_eps : escala de la innovacion en cada regimen. Permite que los
                     regimenes difieran tambien en dispersion y no solo en
                     persistencia.

    Desplazamiento de nivel por regimen
        desplazamientos          : magnitud del desplazamiento d_j. La
                                   separacion entre desplazamientos es lo que
                                   induce multimodalidad en la ley condicional;
                                   con todos iguales, los regimenes se
                                   distinguen unicamente por su dinamica.
        forma_desplazamiento_fn  : forma tau -> f(tau) del desplazamiento,
                                   normalizada a maximo unitario. Por defecto
                                   constante, de modo que d_j es un
                                   desplazamiento vertical uniforme.

    Innovacion funcional
        ell : longitud de correlacion de la innovacion, comun a los regimenes
              para que la diferencia entre ellos resida en la dinamica y la
              escala y no en la suavidad.

    Mecanismo de transicion
        mecanismo         : "probit", "markov" o "quiebre".
        umbrales          : puntos de corte del probit ordenado, en orden
                            creciente y de longitud J - 1. Solo para "probit".
        nitidez           : pendiente del probit. Valores grandes aproximan un
                            umbral duro; valores moderados producen una mezcla
                            genuina con pesos dependientes del estado. Debe ser
                            positivo. Solo para "probit".
        direccion_fn      : direccion e(tau) sobre la cual se proyecta el
                            estado rezagado. Por defecto constante. Solo para
                            "probit".
        matriz_transicion : matriz (J, J) de la cadena oculta, con filas que
                            suman uno. Solo para "markov".
        fracciones_quiebre: fracciones crecientes en (0, 1) de la serie
                            retenida en las que cambia el regimen; su longitud
                            debe ser J - 1. Solo para "quiebre".
    """

    # Operadores y escalas por regimen
    gammas: Sequence[float] = (0.30, 0.30)
    hs_norms: Sequence[float] = (0.30, 0.85)
    sigmas_eps: Sequence[float] = (1.0, 1.0)
    desplazamientos: Sequence[float] = (-1.5, 1.5)
    forma_desplazamiento_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # Innovacion
    ell: float = 0.2

    # Mecanismo de transicion
    mecanismo: str = "probit"
    umbrales: Sequence[float] = (0.0,)
    nitidez: float = 2.0
    direccion_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    matriz_transicion: Optional[np.ndarray] = None
    fracciones_quiebre: Sequence[float] = (0.5,)

    @property
    def n_regimenes(self) -> int:
        return len(self.hs_norms)

    def validar(self) -> None:
        super().validar()

        J = len(self.hs_norms)
        if J < 2:
            raise ValueError(
                "Se requieren al menos dos regimenes; con uno solo el escenario "
                "coincide con el Escenario 1."
            )
        for nombre, secuencia in (
            ("gammas", self.gammas),
            ("sigmas_eps", self.sigmas_eps),
            ("desplazamientos", self.desplazamientos),
        ):
            if len(secuencia) != J:
                raise ValueError(
                    f"{nombre} tiene longitud {len(secuencia)} y hs_norms {J}: "
                    "todas las tuplas por regimen deben tener igual longitud."
                )

        for j, (g, h, s) in enumerate(
            zip(self.gammas, self.hs_norms, self.sigmas_eps)
        ):
            if not (0.0 < h < 1.0):
                raise ValueError(
                    f"hs_norms[{j}]={h}: debe estar en (0, 1) para que el "
                    "regimen sea contractivo."
                )
            if g <= 0:
                raise ValueError(f"gammas[{j}]={g}: debe ser positivo.")
            if s <= 0:
                raise ValueError(f"sigmas_eps[{j}]={s}: debe ser positivo.")

        if self.ell <= 0:
            raise ValueError("ell debe ser positivo.")

        if self.mecanismo not in _MECANISMOS:
            raise ValueError(
                f"mecanismo='{self.mecanismo}' invalido. "
                f"Opciones: {list(_MECANISMOS)}."
            )

        if self.mecanismo == "probit":
            if len(self.umbrales) != J - 1:
                raise ValueError(
                    f"umbrales tiene longitud {len(self.umbrales)}; con J={J} "
                    f"regimenes se requieren {J - 1} puntos de corte."
                )
            if J > 2 and not np.all(np.diff(np.asarray(self.umbrales)) > 0):
                raise ValueError(
                    "umbrales debe ser estrictamente creciente para que el "
                    "probit ordenado asigne probabilidades no negativas."
                )
            if self.nitidez <= 0:
                raise ValueError(
                    "nitidez debe ser positivo. Para un mecanismo independiente "
                    "del estado use mecanismo='markov' con filas identicas."
                )

        elif self.mecanismo == "markov":
            if self.matriz_transicion is None:
                raise ValueError(
                    "mecanismo='markov' requiere matriz_transicion (J, J)."
                )
            P = np.asarray(self.matriz_transicion, dtype=float)
            if P.shape != (J, J):
                raise ValueError(
                    f"matriz_transicion tiene forma {P.shape}; se esperaba ({J}, {J})."
                )
            if np.any(P < 0):
                raise ValueError("matriz_transicion no admite entradas negativas.")
            if not np.allclose(P.sum(axis=1), 1.0):
                raise ValueError("Las filas de matriz_transicion deben sumar uno.")

        else:  # "quiebre"
            f = np.asarray(self.fracciones_quiebre, dtype=float)
            if f.size != J - 1:
                raise ValueError(
                    f"fracciones_quiebre tiene longitud {f.size}; con J={J} "
                    f"regimenes se requieren {J - 1} instantes de quiebre."
                )
            if np.any(f <= 0) or np.any(f >= 1):
                raise ValueError("fracciones_quiebre debe estar contenida en (0, 1).")
            if f.size > 1 and not np.all(np.diff(f) > 0):
                raise ValueError("fracciones_quiebre debe ser estrictamente creciente.")


# ==========================================================================
# MECANISMO DE TRANSICION
# ==========================================================================

def _probabilidades_probit(
    z: float, umbrales: np.ndarray, nitidez: float, J: int
) -> np.ndarray:
    """
    Probabilidades de regimen del probit ordenado evaluado en el estado z:

        P(S <= j | z) = Phi( nitidez * (umbral_j - z) ),  j = 1, ..., J - 1,

    de modo que P(S = j) es la diferencia entre acumuladas consecutivas. Al
    crecer `nitidez` la asignacion tiende a la particion determinista del
    espacio de estados definida por los umbrales.
    """
    acumuladas = np.empty(J + 1)
    acumuladas[0] = 0.0
    acumuladas[J] = 1.0
    acumuladas[1:J] = norm.cdf(nitidez * (umbrales - z))
    probs = np.diff(acumuladas)
    return np.clip(probs, 0.0, None) / max(float(np.sum(np.clip(probs, 0.0, None))), 1e-300)


def _secuencia_quiebre(T: int, fracciones: np.ndarray, J: int) -> np.ndarray:
    """Secuencia determinista de regimenes para el mecanismo de quiebre."""
    cortes = np.floor(np.asarray(fracciones, dtype=float) * T).astype(int)
    s = np.zeros(T, dtype=int)
    for j, corte in enumerate(cortes):
        s[corte:] = j + 1
    return np.clip(s, 0, J - 1)


# ==========================================================================
# DINAMICA DEL ESCENARIO
# ==========================================================================

def simular_trayectoria_far_regimen(
    Psis: np.ndarray,
    chols_K: np.ndarray,
    desplazamientos: np.ndarray,
    mu: np.ndarray,
    cfg: ConfigEscenario3,
    direccion: np.ndarray,
    w_quad: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Itera la recursion autorregresiva con cambio de regimen y devuelve las T
    curvas retenidas SIN ruido de medicion, de dimension (T, L), junto con la
    secuencia de regimenes de esos mismos instantes.

    El estado latente es Y_t = X_t - mu, de modo que la proyeccion que gobierna
    el mecanismo probit se calcula sobre la curva centrada respecto de la media
    global y no respecto del nivel del regimen; con ello el umbral conserva una
    lectura unica a lo largo de toda la serie. El desplazamiento del regimen se
    suma al construir la curva observada.

    Parametros
    ----------
    Psis            : (J, L, L) operadores por regimen.
    chols_K         : (J, L, L) factores de la covarianza de innovacion por regimen.
    desplazamientos : (J, L) desplazamiento de nivel por regimen.
    """
    J, L, _ = Psis.shape
    innovaciones = [generador_innovacion(chols_K[j], rng) for j in range(J)]
    umbrales = np.asarray(cfg.umbrales, dtype=float)
    P_markov = (
        np.asarray(cfg.matriz_transicion, dtype=float)
        if cfg.mecanismo == "markov" else None
    )
    s_quiebre = (
        _secuencia_quiebre(cfg.T, np.asarray(cfg.fracciones_quiebre), J)
        if cfg.mecanismo == "quiebre" else None
    )

    def _sortear(y_prev: np.ndarray, s_prev: int, t_retenido: Optional[int]) -> int:
        if cfg.mecanismo == "probit":
            z = float(np.sum(w_quad * direccion * y_prev))
            return int(rng.choice(J, p=_probabilidades_probit(z, umbrales, cfg.nitidez, J)))
        if cfg.mecanismo == "markov":
            return int(rng.choice(J, p=P_markov[s_prev]))
        # "quiebre": durante el calentamiento rige el primer regimen
        return 0 if t_retenido is None else int(s_quiebre[t_retenido])

    # Calentamiento: la trayectoria se inicializa con una innovacion del
    # primer regimen y se descartan los primeros burn_in periodos.
    s = 0
    Y = innovaciones[0]()
    for _ in range(cfg.burn_in):
        s = _sortear(Y, s, None)
        Y = Psis[s] @ Y + innovaciones[s]()

    curvas = np.empty((cfg.T, L))
    regimenes = np.empty(cfg.T, dtype=int)
    for t in range(cfg.T):
        s = _sortear(Y, s, t)
        Y = Psis[s] @ Y + innovaciones[s]()
        regimenes[t] = s
        curvas[t] = mu + desplazamientos[s] + Y
    return curvas, regimenes


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_3(cfg: ConfigEscenario3) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario 3.

    Los operadores por regimen y las factorizaciones de las covarianzas de
    innovacion se construyen una sola vez y se comparten entre replicas, dado
    que no dependen de la realizacion aleatoria. Cada replica recibe una
    semilla derivada de la semilla maestra, de modo que la replica r es
    identica con independencia de cuantas replicas se generen o del orden de
    ejecucion.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    w_quad = pesos_trapezoidales(tau)
    mu = evaluar_media(cfg.media_fn, tau)
    direccion = _normalizar_direccion(cfg.direccion_fn, tau)
    forma_d = _evaluar_forma(cfg.forma_desplazamiento_fn, tau)

    J = cfg.n_regimenes
    Psis = np.empty((J, cfg.L, cfg.L))
    Ks = np.empty((J, cfg.L, cfg.L))
    chols_K = np.empty((J, cfg.L, cfg.L))
    desplazamientos = np.empty((J, cfg.L))

    for j in range(J):
        Psis[j] = matriz_operador_ar(tau, cfg.gammas[j], cfg.hs_norms[j])
        Ks[j] = matriz_covarianza_innovacion(tau, cfg.sigmas_eps[j], cfg.ell)
        chols_K[j] = factor_cholesky(Ks[j], cfg.jitter)
        desplazamientos[j] = cfg.desplazamientos[j] * forma_d

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    regimenes = np.empty((cfg.R, cfg.T), dtype=int)

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        curvas_r, regimenes_r = simular_trayectoria_far_regimen(
            Psis, chols_K, desplazamientos, mu, cfg, direccion, w_quad, rng
        )
        curvas[r] = curvas_r
        regimenes[r] = regimenes_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={
            "operadores": Psis,
            "cov_innovaciones": Ks,
            "desplazamientos": desplazamientos,
            "direccion_estado": direccion,
            "regimenes": regimenes,
            "pesos_cuadratura": w_quad,
        },
    )
    salida.diagnostico = resumen_escenario_3(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def resumen_escenario_3(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con las verificaciones propias del Algoritmo 3.

    Contractividad. Se reporta la norma de Hilbert-Schmidt efectiva de cada
    regimen y su maximo; que este sea menor que uno es la condicion suficiente
    de ergodicidad de los mecanismos "probit" y "markov". El mecanismo
    "quiebre" es no estacionario por construccion y se reporta como tal con
    independencia de las normas.

    Ocupacion de los regimenes. Se reportan la proporcion de instantes en cada
    regimen y la duracion media de las rachas. Un regimen con ocupacion
    marginal no aporta informacion al ajuste por escasez de observaciones, y
    rachas de duracion cercana a uno indican una alternancia tan rapida que la
    mezcla resulta indistinguible de un unico regimen con mayor varianza.

    Separacion de los regimenes. Se reporta la distancia maxima en L^2 entre
    los desplazamientos de nivel, normalizada por la desviacion estandar
    puntual del proceso. Valores pequenos indican regimenes superpuestos, en
    los cuales la multimodalidad de la ley condicional no es detectable y el
    escenario pierde su capacidad de discriminar entre metodos.
    """
    if not isinstance(salida.config, ConfigEscenario3):
        raise TypeError(
            "resumen_escenario_3 requiere una salida generada con "
            f"ConfigEscenario3; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config

    Psis = salida.internos.get("operadores")
    w_quad = salida.internos.get("pesos_cuadratura")
    regimenes = salida.internos.get("regimenes")
    desplazamientos = salida.internos.get("desplazamientos")
    for nombre, objeto in (
        ("operadores", Psis), ("pesos_cuadratura", w_quad),
        ("regimenes", regimenes), ("desplazamientos", desplazamientos),
    ):
        if objeto is None:
            raise KeyError(
                f"La salida no contiene '{nombre}' en `internos`; no puede "
                "completarse el control de calidad del Escenario 3."
            )

    J = Psis.shape[0]
    hs = np.array([norma_hilbert_schmidt(Psis[j], w_quad) for j in range(J)])
    radios = np.array([float(np.max(np.abs(np.linalg.eigvals(Psis[j])))) for j in range(J)])

    # Ocupacion y persistencia de los regimenes
    R, T = regimenes.shape
    proporciones = np.array(
        [float(np.mean(regimenes == j)) for j in range(J)]
    )
    transiciones = np.mean([int(np.sum(np.diff(regimenes[r]) != 0)) for r in range(R)])
    duracion_media = float(T / (transiciones + 1.0))

    duraciones_por_regimen = np.full(J, np.nan)
    for j in range(J):
        rachas = []
        for r in range(R):
            en_j = (regimenes[r] == j).astype(int)
            if en_j.sum() == 0:
                continue
            bordes = np.diff(np.concatenate(([0], en_j, [0])))
            inicios = np.flatnonzero(bordes == 1)
            finales = np.flatnonzero(bordes == -1)
            rachas.extend((finales - inicios).tolist())
        if rachas:
            duraciones_por_regimen[j] = float(np.mean(rachas))

    # Separacion de niveles entre regimenes, en unidades de desviacion puntual
    sd_puntual = float(np.sqrt(max(base["var_puntual_media"], 1e-300)))
    dist_L2 = np.zeros((J, J))
    for j in range(J):
        for k in range(J):
            dif = desplazamientos[j] - desplazamientos[k]
            dist_L2[j, k] = float(np.sqrt(np.sum(w_quad * dif ** 2)))
    separacion_max = float(dist_L2.max())

    estacionario = bool(hs.max() < 1.0 and cfg.mecanismo != "quiebre")

    especifico = {
        "mecanismo": cfg.mecanismo,
        "n_regimenes": int(J),
        "hs_norm_objetivo": [float(h) for h in cfg.hs_norms],
        "hs_norm_efectiva": [float(h) for h in hs],
        "hs_norm_error_absoluto_max": float(
            np.max(np.abs(hs - np.asarray(cfg.hs_norms, dtype=float)))
        ),
        "hs_norm_maxima": float(hs.max()),
        "radio_espectral_operadores": [float(x) for x in radios],
        "contractividad_todos_los_regimenes": bool(hs.max() < 1.0),
        "estacionariedad_garantizada": estacionario,
        "proporcion_regimen": [float(p) for p in proporciones],
        "proporcion_regimen_minima": float(proporciones.min()),
        "n_transiciones_media": float(transiciones),
        "duracion_media_racha": duracion_media,
        "duracion_media_por_regimen": [float(d) for d in duraciones_por_regimen],
        "separacion_niveles_L2_max": separacion_max,
        "separacion_en_sd_puntual": float(separacion_max / sd_puntual),
    }
    return {**base, **especifico}
