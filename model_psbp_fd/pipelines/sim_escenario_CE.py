"""
sim_escenario_CE.py
====================
Escenario CE: COMPOSICION DE ESTADOS sobre los coeficientes de una base
ortonormal fija. Escenario de diagnostico, no es un Algoritmo del anexo. Se
nombra CE y no con una letra suelta porque las letras libres del abecedario ya
colisionan con simbolos del proyecto (M componentes, N atomos, L grilla,
K funciones base, G grilla de localizacion).

    corrida 46 : cuatro fases, base Fourier d = 11, disparo por umbral

El estado NO es un juego de matrices de regresion que conmutan --eso es el
Escenario L de las corridas 41 a 45-- sino una LEY DE GENERACION por estado:

    c_t | estado ~ N( mu(estado), Sigma(estado) )

con cuatro estados encadenados en un ciclo, y dos de ellos son TRANSICIONES en
las que mu y Sigma no son constantes sino que migran de un extremo al otro:

    BASE   (0)  mu_1, Sigma_1.  Es la normalidad. Se queda aqui hasta que la
                regla dispara: z_t = <w, c_{t-1}> / sd_1  supera  c_in.
    SUBIDA (1)  mu y Sigma interpolan LINEALMENTE de (mu_1, Sigma_1) a
                (mu_2, Sigma_2) a lo largo de D_1 periodos. Es la tendencia.
    MESETA (2)  mu_2, Sigma_2 durante D_2 periodos. El nivel se queda donde
                llego; la innovacion sigue corriendo, de modo que el tramo es
                plano en la media y no una serie muerta.
    BAJADA (3)  mu y Sigma vuelven de (mu_2, Sigma_2) a (mu_1, Sigma_1) con
                perfil EXPONENCIAL a lo largo de D_3 periodos, y el ciclo
                vuelve a BASE.

Cada `D_i` se sortea al entrar en la fase con `D = d_min + Poisson(lambda_i)` y
la fase dura exactamente eso: la regla no se vuelve a mirar hasta que la
duracion se cumple. Esa es la parte "sigue determinada distribucion hasta que
se cumpla".

Por que la tendencia no es un estado mas con su (mu_3, Sigma_3)
----------------------------------------------------------------
Porque un estado con media constante genera un tramo PLANO, no una tendencia.
Para que la trayectoria suba hay que mover la media DENTRO de la fase: en el
periodo k de una subida de duracion D_1,

    mu_t = mu_1 + (k / D_1) (mu_2 - mu_1),    Sigma_t igual con la misma
                                              fraccion sobre la desviacion

de modo que la curva recorre el trayecto completo entre los dos niveles en
exactamente D_1 periodos, con la pendiente que esa duracion imponga. Esa es la
correccion: los coeficientes de una fase de tendencia se generan con una ley
que cambia en cada periodo, mientras que los de BASE y MESETA se generan con
una ley fija.

El perfil de la bajada es `frac(k) = (tol^(k/D_3) - tol) / (1 - tol)`, que vale
1 al empezar y EXACTAMENTE 0 al terminar: es un decaimiento exponencial de
razon `tol^(1/D_3)` re-escalado para que cierre en el nivel de partida en vez
de dejar un residuo del `tol` por ciento. Sin ese re-escalado el ciclo
acumularia un resto en cada episodio y "volver a la normalidad" seria
aproximado.

La correccion de la innovacion
-------------------------------
Dentro de un estado la ley marginal tiene que ser la que el estado declara. Con

    c_t = mu_t + phi (c_{t-1} - mu_{t-1}) + eps_t,
    eps_t ~ N(0, (1 - phi^2) diag(sigma_t)^2)

el factor `(1 - phi^2)` es justamente el que hace que la varianza estacionaria
valga `sigma_t^2` y no `sigma_t^2 / (1 - phi^2)`. Sin el, subir la persistencia
cambiaria a la vez la memoria y la dispersion de cada estado, y `Sigma_1` y
`Sigma_2` dejarian de ser las que la configuracion dice que son.

`phi` no es decoracion, y su valor se midio en vez de elegirse. Con `phi = 0`
los coeficientes son independientes dado el estado y la unica senyal predecible
es el estado mismo; con `phi` alto el rezago predice tan bien que el estado deja
de importar. R^2 fuera de muestra en scores con M = 3, seis replicas, todo lo
demas en los valores de la corrida:

    phi     lineal   techo por fase   ganancia
    0.00     0.168       0.266         +0.097
    0.20     0.277       0.373         +0.096
    0.35     0.347       0.396         +0.049
    0.50     0.469       0.516         +0.047

Con 0.50 --el valor que se probo primero-- la referencia lineal ya recupera casi
todo y la ganancia por conocer la fase cae a la mitad. Con 0.20 la ganancia se
mantiene en su maximo y el lineal conserva 0.28 de R^2, que es lo que impide que
el escenario sea atacable como construido para ganar: el FAR tiene senyal real
que recuperar. Ese es el valor de la corrida.

Que tiene que capturar el modelo
----------------------------------
La media condicional verdadera es no lineal y depende de un estado que el
pasado observable no revela: viendo `c_{t-1}, c_{t-2}, c_{t-3}` no se sabe
cuantos periodos le quedan a la fase en curso. La predictiva es entonces una
mezcla,

    p(c_t | pasado) = sum_s P(estado_t = s | pasado) N(mu_s(.), Sigma_s(.)),

y a diferencia del Escenario L esa mezcla tiene componentes con VARIANZAS
distintas --`Sigma_2 = razon_sigma^2 Sigma_1`-- ademas de medias distintas. Ahi
el Bloque B tiene algo que medir: una banda de ancho condicionalmente constante
no puede cubrir bien a la vez el estado de dispersion baja y el de dispersion
alta.

Medido con la configuracion de la corrida, la varianza del residuo de la fase
mas agitada es 2.2 a 2.6 veces la de la mas tranquila
(`razon_var_condicional_entre_fases`). Con `razon_sigma = 1` esa razon cae a
1.4 --lo que queda es solo el efecto de las fases de tendencia sobre la media--
y el FAR recupera la forma de banda correcta: ese es el control natural de esta
corrida si alguna vez hace falta separar "gana por la media" de "gana por la
dispersion".

La parte que este escenario NO ofrece: la ganancia por conocer la fase no viene
de que la fase sea irrecuperable. Un discriminante lineal sobre los tres rezagos
la acierta el 68 % de las veces contra un 36 % de la clase mayoritaria, porque
el nivel dice bastante sobre en que parte del ciclo se esta. Lo que no puede
saber es cuantos periodos le quedan a la fase en curso, y de ahi sale la mezcla.

La escala del indice de disparo es analitica
----------------------------------------------
En el Escenario L, `sd(z)` dependia de la dinamica, que dependia del regimen,
que dependia de `sd(z)`, y habia que resolver un punto fijo iterando el
calentamiento. Aqui no: en el estado BASE la ley es exactamente
`N(mu_1, Sigma_1)`, de modo que

    sd(z) = sqrt( w^T Sigma_1 w )

en forma cerrada, y `c_in` significa literalmente "tantas desviaciones del
estado base". Es una simplificacion real del diseno, no un atajo.

Por que base Fourier y por que el salto va en la coordenada constante
-----------------------------------------------------------------------
`base_ortonormal` se importa del Escenario L y admite las dos bases. Se usa
"fourier" (que exige `d` impar, de ahi `d = 11`) porque su primera funcion es
la CONSTANTE: cargar el salto en la coordenada 0 hace que el episodio sea un
desplazamiento vertical de la curva entera, que es la lectura visual que el
escenario busca --la curva sube, se queda arriba, baja y vuelve--. Con la
B-spline de Lowdin el salto seria local, una region del dominio que sube
mientras el resto no, que es otra cosa.

El precio a declarar: la base Fourier no es local, de modo que la lectura
espacial "una region gobierna a otra" que tenian las corridas 41-45 aqui no
existe. A cambio, `cond(W) = 1` exacto y el nivel es una coordenada.

`perfil_salto` permite mover el salto a cualquier direccion --por ejemplo dar
al estado ALTO una forma y no solo un nivel-- y `mu_bajo` desplazar el estado
base. Por defecto son la constante y el cero.

Con `d = 11` y esta dinamica, el FPCA del pipeline retiene el 89.0 % de la
varianza en la primera componente y el 97.8 % en dos, de modo que la regla del
95 % cae en `M = 2` y el barrido `(1, 2)` de la corrida 45 la rodea sin
cambiarlo. La primera componente es practicamente el nivel, que es donde vive
el episodio.

Calibracion de los parametros del mecanismo
---------------------------------------------
`c_in` y las tres lambdas fijan el reparto de las cuatro fases, y el reparto es
la restriccion dura: el pipeline exige que ninguna baje de 0.15 porque una fase
casi vacia no tiene datos con que estimar su ley. Barrido sobre seis replicas:

    c_in  d_min  lam(s/m/b)   base/subida/meseta/bajada   episodios  ganancia
    1.00    2      3/4/3       .32  .22  .25  .21            43       +0.096
    1.00    3      4/5/4       .34  .21  .24  .21            30       +0.025
    1.25    2      3/4/3       .42  .18  .22  .19            36       +0.065
    1.50    2      3/4/3       .53  .15  .17  .15            29       -0.007

Subir `c_in` engorda el estado base a costa de las otras tres y de los
episodios; alargar las fases reduce cuantos ciclos caben en T = 1000, y con 30
episodios en la serie completa quedan menos de diez en el bloque de prueba, que
es poco para que la ventana movil los vea. La configuracion de la corrida
--`c_in = 1.0`, `d_min = 2`, lambdas 3/4/3-- deja 42 episodios, ninguna fase por
debajo de 0.20 y la ganancia en su maximo.

`salto` fija cuanto se ve el episodio: con 3.0 la amplitud es 1.4 desviaciones
de la curva y 12 veces el ruido de observacion, de modo que el mecanismo es
visible a ojo y no algo que solo aparece en las metricas.

Author: model_psbp_fd
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .sim_comun import (
    ConfigObservacion,
    SalidaSimulacion,
    aplicar_ruido_observacion,
    diagnostico_comun,
    evaluar_media,
    grilla_regular,
    semillas_replicas,
)
# La base ortonormal y su verificacion son las del Escenario L: el mecanismo es
# lo unico nuevo, y duplicar la construccion de la base haria que las dos
# familias pudieran divergir sin que nadie lo note.
from .sim_escenario_L import base_ortonormal
from ..utils.quadrature import pesos_trapezoidales

__all__ = [
    "ConfigEscenarioCE",
    "FASES",
    "generar_escenario_CE",
    "resumen_escenario_CE",
    "simular_coeficientes_CE",
]

BASE, SUBIDA, MESETA, BAJADA = 0, 1, 2, 3
FASES = {BASE: "base", SUBIDA: "subida", MESETA: "meseta", BAJADA: "bajada"}
SIGUIENTE = {BASE: SUBIDA, SUBIDA: MESETA, MESETA: BAJADA, BAJADA: BASE}


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioCE(ConfigObservacion):
    """
    Parametros del Escenario CE.

    Hereda de `ConfigObservacion` el esquema de observacion (L, T, burn_in,
    sigma_obs, R, seed, media_fn, jitter), identico al de las corridas vivas.

    Base generadora
        d             : numero de coeficientes. Con "fourier" debe ser impar.
        base          : "fourier" o "bspline_lowdin".
        orden_bspline : orden previo a la ortonormalizacion de Lowdin; inerte
                        con "fourier".

    Los dos estados con ley fija
        mu_bajo      : media de los coeficientes en el estado BASE. None = 0.
        salto        : distancia entre las dos medias, en unidades del perfil.
        perfil_salto : direccion del salto, se normaliza a norma 1. None usa
                       e_0, que con base Fourier es la funcion CONSTANTE: el
                       episodio es entonces un desplazamiento vertical de toda
                       la curva.
        sigma_base   : escala de la desviacion del estado BASE.
        decaimiento  : sigma_j = sigma_base * decaimiento^j. Da el espectro
                       decreciente sin el cual el barrido en M no significaria
                       nada.
        razon_sigma  : Sigma del estado MESETA = (razon_sigma)^2 veces la del
                       BASE. Con 1.0 los dos estados solo difieren en media;
                       por encima de 1 difieren tambien en dispersion, que es
                       lo que le da al Bloque B algo que medir.
        phi          : persistencia dentro del estado, en (-1, 1). Ver el
                       docstring del modulo: con 0 la unica senyal predecible
                       es el estado y la comparacion contra el FAR deja de ser
                       informativa.

    Las dos fases de transicion
        d_min        : duracion minima de CUALQUIER fase, en periodos.
        lambda_subida / lambda_meseta / lambda_bajada :
                       media de la Poisson que se suma a d_min en cada fase.
                       Controlan la forma del trapecio: cuanto dura la rampa,
                       cuanto la meseta y cuanto el retorno.
        tol_retorno  : fraccion del salto que queda cuando la exponencial de
                       bajada llega al ultimo periodo, ANTES del re-escalado
                       que la cierra en cero. Gobierna la curvatura: cerca de 1
                       la bajada es casi recta, cerca de 0 cae de golpe y se
                       arrastra.

    La regla de disparo
        w    : direccion del indice de disparo. None usa e_0, la misma
               coordenada del salto: el episodio arranca cuando el nivel del
               estado base sube por azar.
        c_in : umbral, en desviaciones del estado BASE. Es exacto y no
               calibrado: sd(z) = sqrt(w^T Sigma_1 w) en forma cerrada.

    Solo diagnostico: `m_diagnostico` y `prop_train_diag` replican las
    condiciones del pipeline para medir donde vive el modelo.
    """

    # --- Base generadora ---
    d: int = 11
    base: str = "fourier"
    orden_bspline: int = 4

    # --- Los dos estados con ley fija ---
    mu_bajo: Optional[np.ndarray] = None
    salto: float = 3.0
    perfil_salto: Optional[np.ndarray] = None
    sigma_base: float = 1.0
    decaimiento: float = 0.45
    razon_sigma: float = 1.6
    phi: float = 0.2

    # --- Fases ---
    d_min: int = 2
    lambda_subida: float = 3.0
    lambda_meseta: float = 4.0
    lambda_bajada: float = 3.0
    tol_retorno: float = 0.15

    # --- Regla de disparo ---
    w: Optional[np.ndarray] = None
    c_in: float = 1.00

    # --- Solo diagnostico ---
    m_diagnostico: int = 3
    prop_train_diag: float = 0.70

    def validar(self) -> None:
        super().validar()
        if self.d < 2:
            raise ValueError("d debe ser al menos 2.")
        if self.base not in {"bspline_lowdin", "fourier"}:
            raise ValueError(
                f"base={self.base!r}: debe ser 'bspline_lowdin' o 'fourier'."
            )
        if self.base == "fourier" and self.d % 2 == 0:
            raise ValueError(
                f"base='fourier' exige d impar; se recibio d={self.d}. Use d "
                "impar o cambie a 'bspline_lowdin', que admite cualquier d."
            )
        if self.salto == 0:
            raise ValueError(
                "salto = 0: los dos estados tendrian la misma media y no habria "
                "episodio que ver."
            )
        if self.sigma_base <= 0:
            raise ValueError("sigma_base debe ser positivo.")
        if not (0.0 < self.decaimiento <= 1.0):
            raise ValueError("decaimiento debe estar en (0, 1].")
        if self.razon_sigma <= 0:
            raise ValueError("razon_sigma debe ser positivo.")
        if not (-1.0 < self.phi < 1.0):
            raise ValueError(
                f"phi={self.phi}: debe estar en (-1, 1). Es la persistencia "
                "dentro del estado y con |phi| >= 1 la serie no es estacionaria "
                "ni siquiera con el estado fijo."
            )
        if self.d_min < 1:
            raise ValueError(
                f"d_min={self.d_min}: debe ser al menos 1, de modo que toda "
                "fase dure al menos un periodo. Una fase de duracion 0 saltaria "
                "de la subida a la bajada sin meseta."
            )
        for nombre in ("lambda_subida", "lambda_meseta", "lambda_bajada"):
            if getattr(self, nombre) < 0:
                raise ValueError(f"{nombre} no puede ser negativo.")
        if not (0.0 < self.tol_retorno < 1.0):
            raise ValueError(
                f"tol_retorno={self.tol_retorno}: debe estar en (0, 1). Es la "
                "fraccion del salto que quedaria al final de la exponencial "
                "antes del re-escalado."
            )
        if self.c_in <= 0:
            raise ValueError("c_in debe ser positivo.")
        for nombre in ("mu_bajo", "perfil_salto", "w"):
            v = getattr(self, nombre)
            if v is not None and np.asarray(v).shape != (self.d,):
                raise ValueError(
                    f"{nombre} debe tener forma ({self.d},); se recibio "
                    f"{np.asarray(v).shape}."
                )

    def to_dict(self) -> dict:
        """Como la de la clase base, pero deja los vectores serializables."""
        d = super().to_dict()
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


# ==========================================================================
# LEY DE CADA FASE
# ==========================================================================

def _sortear_duracion(d_min: int, lam: float, rng: np.random.Generator) -> int:
    """`D = d_min + Poisson(lambda)`. Unico punto a tocar para cambiar de familia."""
    return int(d_min + rng.poisson(lam))


def _fraccion(fase: int, k: int, D: int, tol: float) -> float:
    """
    Posicion entre los dos estados: 0 = BASE, 1 = MESETA.

    La subida es lineal en `k/D`, de modo que la curva recorre el salto entero
    en D periodos con pendiente constante. La bajada es exponencial re-escalada
    para cerrar EXACTAMENTE en 0 en el ultimo periodo; sin el re-escalado cada
    episodio dejaria un residuo del `tol` por ciento del salto.
    """
    if fase == BASE:
        return 0.0
    if fase == MESETA:
        return 1.0
    if fase == SUBIDA:
        return min(1.0, k / max(D, 1))
    razon = tol ** (min(k, D) / max(D, 1))
    return float((razon - tol) / (1.0 - tol))


def _ley(frac: float, mu_bajo: np.ndarray, delta_mu: np.ndarray,
         sigma_bajo: np.ndarray, sigma_alto: np.ndarray):
    """
    `(mu_t, sigma_t)` de un periodo, a partir de su posicion entre los estados.

    La interpolacion de la dispersion se hace sobre la DESVIACION y no sobre la
    varianza: asi la transicion en escala es la misma que la de la media y la
    fase de tendencia no tiene un tramo en que la dispersion cambie mas rapido
    que el nivel.
    """
    return (mu_bajo + frac * delta_mu,
            sigma_bajo + frac * (sigma_alto - sigma_bajo))


# ==========================================================================
# DINAMICA
# ==========================================================================

def simular_coeficientes_CE(cfg: ConfigEscenarioCE, mu_bajo, delta_mu,
                            sigma_bajo, sigma_alto, w, sd_z,
                            rng: np.random.Generator):
    """
    Itera la composicion de estados y devuelve `(C, S, FRAC, disparos)`.

    C (T, d) los coeficientes, S (T,) la fase en curso --0 base, 1 subida,
    2 meseta, 3 bajada--, FRAC (T,) la posicion entre los dos estados y
    `disparos` (T,) booleano marcando el periodo en que la regla arranca un
    episodio.

    La grilla no interviene: el estado es el vector de coeficientes mas el
    trio (fase, periodos transcurridos, duracion sorteada). El calentamiento
    arranca en BASE y `burn_in` lo descarta.
    """
    d = cfg.d
    lambdas = {SUBIDA: cfg.lambda_subida, MESETA: cfg.lambda_meseta,
               BAJADA: cfg.lambda_bajada}

    c_prev = mu_bajo + sigma_bajo * rng.normal(size=d)
    mu_prev = mu_bajo.copy()
    fase, k, D = BASE, 0, 0

    n_total = cfg.burn_in + cfg.T
    C = np.empty((n_total, d))
    S = np.empty(n_total, dtype=int)
    FRAC = np.empty(n_total)
    DIS = np.zeros(n_total, dtype=bool)

    for t in range(n_total):
        z = float(w @ (c_prev - mu_bajo)) / sd_z

        if fase == BASE:
            if z > cfg.c_in:                    # la regla dispara el episodio
                fase, k, D = SUBIDA, 1, _sortear_duracion(cfg.d_min,
                                                          lambdas[SUBIDA], rng)
                DIS[t] = True
        else:
            k += 1
            if k > D:                           # la duracion se cumplio
                fase = SIGUIENTE[fase]
                if fase == BASE:
                    k, D = 0, 0
                else:
                    k, D = 1, _sortear_duracion(cfg.d_min, lambdas[fase], rng)

        frac = _fraccion(fase, k, D, cfg.tol_retorno)
        mu_t, sigma_t = _ley(frac, mu_bajo, delta_mu, sigma_bajo, sigma_alto)

        # (1 - phi^2) es lo que hace que la marginal dentro del estado sea
        # exactamente N(mu_t, diag(sigma_t)^2). Ver el docstring del modulo.
        eps = np.sqrt(1.0 - cfg.phi ** 2) * sigma_t * rng.normal(size=d)
        c = mu_t + cfg.phi * (c_prev - mu_prev) + eps

        C[t], S[t], FRAC[t] = c, fase, frac
        c_prev, mu_prev = c, mu_t

    if not np.all(np.isfinite(C)):
        raise FloatingPointError(
            "La trayectoria de coeficientes no es finita. Con |phi| < 1 y "
            "duraciones finitas esto no deberia ocurrir: revise sigma_base y "
            "razon_sigma."
        )
    b = cfg.burn_in
    return C[b:], S[b:], FRAC[b:], DIS[b:]


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_CE(cfg: ConfigEscenarioCE) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario CE.

    La base, las dos leyes de estado y la escala del indice de disparo no
    dependen de la realizacion y se construyen una sola vez.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    mu_curva = evaluar_media(cfg.media_fn, tau)
    Phi, info_base = base_ortonormal(tau, cfg.d, cfg.base, cfg.orden_bspline)

    mu_bajo = (np.zeros(cfg.d) if cfg.mu_bajo is None
               else np.asarray(cfg.mu_bajo, dtype=float).copy())

    if cfg.perfil_salto is None:
        perfil = np.zeros(cfg.d)
        perfil[0] = 1.0                 # con base Fourier, la funcion constante
    else:
        perfil = np.asarray(cfg.perfil_salto, dtype=float).copy()
    perfil = perfil / np.linalg.norm(perfil)
    delta_mu = cfg.salto * perfil

    sigma_bajo = cfg.sigma_base * cfg.decaimiento ** np.arange(cfg.d, dtype=float)
    sigma_alto = cfg.razon_sigma * sigma_bajo

    if cfg.w is None:
        w = np.zeros(cfg.d)
        w[0] = 1.0
    else:
        w = np.asarray(cfg.w, dtype=float).copy()
    w = w / np.linalg.norm(w)
    # Cerrada, no calibrada: en BASE la ley es exactamente N(mu_1, Sigma_1).
    sd_z = float(np.sqrt(np.sum((w * sigma_bajo) ** 2)))

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    coeficientes = np.empty((cfg.R, cfg.T, cfg.d))
    fase_t = np.empty((cfg.R, cfg.T), dtype=int)
    fraccion_t = np.empty((cfg.R, cfg.T))
    disparo_t = np.empty((cfg.R, cfg.T), dtype=bool)

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        C, S, FR, DIS = simular_coeficientes_CE(
            cfg, mu_bajo, delta_mu, sigma_bajo, sigma_alto, w, sd_z, rng)
        # Unico punto donde interviene la grilla.
        curvas_r = mu_curva[None, :] + C @ Phi
        curvas[r] = curvas_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)
        coeficientes[r] = C
        fase_t[r] = S
        fraccion_t[r] = FR
        disparo_t[r] = DIS

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu_curva,
        semillas=registro,
        config=cfg,
        internos={
            "base": Phi,
            "info_base": info_base,
            "coeficientes": coeficientes,
            "fase_t": fase_t,
            "fraccion_t": fraccion_t,
            "disparo_t": disparo_t,
            "mu_bajo": mu_bajo,
            "mu_alto": mu_bajo + delta_mu,
            "delta_mu": delta_mu,
            "sigma_bajo": sigma_bajo,
            "sigma_alto": sigma_alto,
            "w": w,
            "sd_z": sd_z,
            "pesos_cuadratura": pesos_trapezoidales(tau),
        },
    )
    salida.diagnostico = resumen_escenario_CE(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def _duraciones(S: np.ndarray, fase: int) -> np.ndarray:
    """Longitudes de las rachas de `fase`, en periodos."""
    m = S == fase
    if not m.any():
        return np.zeros(0, dtype=int)
    corte = np.flatnonzero(np.diff(m.astype(int)) != 0) + 1
    bloques = np.split(np.arange(S.size), corte)
    return np.array([len(b) for b in bloques if m[b[0]]], dtype=int)


def _diseno(C: np.ndarray, M: int, n_lags: int, prop_train: float):
    """
    Diseno AR en el espacio de scores truncado a `M`, que es lo que ven los
    modelos. Con base ortonormal la FPCA en metrica L^2 es el PCA euclideo de
    cov(C), de modo que proyectar sobre sus M primeros autovectores reproduce
    exactamente las covariables del pipeline.
    """
    M = int(min(M, C.shape[1]))
    U = np.linalg.eigh(np.cov(C.T))[1][:, ::-1][:, :M]
    Z = C @ U
    Y = Z[n_lags:]
    X = np.hstack([np.ones((Y.shape[0], 1))]
                  + [Z[n_lags - l: Z.shape[0] - l] for l in range(1, n_lags + 1)])
    corte = int(prop_train * C.shape[0]) - n_lags
    return X, Y, np.arange(corte), np.arange(corte, Y.shape[0])


def _diagnostico_predictivo(C, S, M, n_lags, prop_train):
    """
    R^2 fuera de muestra del mejor predictor LINEAL y del techo alcanzable
    conociendo la fase, ambos en el espacio de scores truncado a `M`.

    El techo es una regresion POR FASE con la fase conocida, no la media
    condicional exacta del generador: es lo que un modelo que ve M componentes
    y tiene que estimar podria alcanzar si ademas se le regalara el estado.
    Reportar el oraculo exacto sobreestimaria lo que el PSBPM-FD puede ganar.

    La diferencia entre ambos decide si el escenario sirve, y el R^2 lineal NO
    ha de ser nulo: si la referencia lineal degenera a la media incondicional,
    el resultado es facil de atacar como construido para ganar.
    """
    X, Y, tr, te = _diseno(C, M, n_lags, prop_train)
    s = S[n_lags:]
    sst = ((Y[te] - Y[tr].mean(0)) ** 2).sum()
    if sst <= 0:
        return float("nan"), float("nan")

    beta, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
    r2_lin = float(1.0 - ((Y[te] - X[te] @ beta) ** 2).sum() / sst)

    P = np.zeros_like(Y)
    for k in np.unique(s):
        idx = np.where(s == k)[0]
        idx_tr = np.intersect1d(tr, idx)
        if idx_tr.size <= X.shape[1]:
            return r2_lin, float("nan")
        bk, *_ = np.linalg.lstsq(X[idx_tr], Y[idx_tr], rcond=None)
        P[idx] = X[idx] @ bk
    return r2_lin, float(1.0 - ((Y[te] - P[te]) ** 2).sum() / sst)


def _diagnostico_mezcla(C, S, M, n_lags, prop_train):
    """
    `(accuracy, accuracy_base, inflacion_media, inflacion_p90, razon_var)`: si
    el mecanismo produjo una mezcla de verdad, y de que tipo.

    `accuracy` es la de un discriminante lineal gaussiano con covarianza comun
    entrenado para predecir la FASE desde los `n_lags` rezagos --exactamente la
    informacion que ven el FAR, los arboles y el PSBPM-FD--. Si fuera ~1 la
    fase seria recuperable y no habria mezcla que capturar; lo que la aleja de 1
    es que el pasado no dice cuantos periodos le quedan a la fase en curso.

    `inflacion` es `1 + Var_s(m_s(x)) / sigma^2(x)` punto a punto, con los pesos
    `P(fase | pasado)` del mismo discriminante: cuanto ensancha la predictiva el
    no saber en que fase se esta. La distancia entre su media y su percentil 90
    es lo que una banda de ancho condicionalmente constante no puede seguir.

    `razon_var` es el cociente entre la mayor y la menor varianza de residuo
    POR FASE. Mide la otra fuente de no gaussianidad del escenario, que la
    inflacion no ve: la dispersion propia de cada estado --`Sigma_2` es
    `razon_sigma^2` veces `Sigma_1`-- de modo que una banda de ancho unico
    sobre-cubre en el estado tranquilo y sub-cubre en el agitado. Es la cifra
    del Bloque B: si vale 1, los estados solo difieren en media y el FAR tiene
    la forma de banda correcta.
    """
    X, Y, tr, te = _diseno(C, M, n_lags, prop_train)
    s = S[n_lags:]
    F = X[:, 1:]
    clases = np.unique(s[tr])
    nan5 = (float("nan"),) * 5
    if clases.size < 2:
        return nan5

    medias, priors = [], []
    Sw = np.zeros((F.shape[1], F.shape[1]))
    for k in clases:
        idx = tr[s[tr] == k]
        if idx.size <= F.shape[1]:
            return nan5
        m = F[idx].mean(0)
        medias.append(m)
        priors.append(idx.size / tr.size)
        Dv = F[idx] - m
        Sw += Dv.T @ Dv
    Sw /= max(tr.size - clases.size, 1)
    Si = np.linalg.pinv(Sw)

    puntajes = np.column_stack([F @ Si @ m - 0.5 * m @ Si @ m + np.log(p)
                                for m, p in zip(medias, priors)])
    acc = float((clases[puntajes[te].argmax(1)] == s[te]).mean())
    acc_base = float((s[te] == np.bincount(s[tr]).argmax()).mean())

    P = np.exp(puntajes - puntajes.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)

    betas = []
    for k in clases:
        idx = tr[s[tr] == k]
        bk, *_ = np.linalg.lstsq(X[idx], Y[idx], rcond=None)
        betas.append(bk)
    medias_s = np.stack([X[te] @ bk for bk in betas])
    pesos = P[te].T[:, :, None]
    centro = (pesos * medias_s).sum(0)
    var_entre = (pesos * (medias_s - centro) ** 2).sum(0)
    res = np.stack([Y[te] - X[te] @ bk for bk in betas])
    sigma2 = (pesos * res ** 2).sum(0).mean(0)
    infl = 1.0 + var_entre / np.where(sigma2 > 0, sigma2, np.nan)

    # Varianza del residuo DENTRO de cada fase, con su propia regresion: el
    # cociente entre la mayor y la menor es la heterocedasticidad que una banda
    # de ancho unico no puede seguir.
    var_fase = []
    for j, k in enumerate(clases):
        idx = te[s[te] == k]
        if idx.size > X.shape[1]:
            var_fase.append(float(((Y[idx] - X[idx] @ betas[j]) ** 2).mean()))
    razon_var = (max(var_fase) / min(var_fase)
                 if len(var_fase) >= 2 and min(var_fase) > 0 else float("nan"))

    return (acc, acc_base, float(np.nanmean(infl)),
            float(np.nanpercentile(infl, 90)), razon_var)


def resumen_escenario_CE(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con lo propio de la composicion de estados. Lo
    que decide si el escenario sirve:

    *   el reparto de las CUATRO fases --una fase casi vacia no tiene datos con
        que estimar su ley, y el pipeline exige que ninguna baje de 0.15--;
    *   `n_episodios`, porque con pocos ciclos completos el bloque de prueba
        puede no contener ninguno y la evaluacion no medir el mecanismo;
    *   `amplitud_episodio_curva`, el salto efectivo en unidades de la
        desviacion de la curva: si es pequenyo, el episodio queda enterrado en
        el ruido de observacion y no hay nada que capturar;
    *   `r2_lineal_en_scores` contra `r2_por_fase_en_scores`;
    *   `accuracy_fase_desde_pasado` e `inflacion_varianza_mezcla_*`.
    """
    if not isinstance(salida.config, ConfigEscenarioCE):
        raise TypeError(
            "resumen_escenario_CE requiere una salida generada con "
            f"ConfigEscenarioCE; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config
    C = salida.internos["coeficientes"]
    S = salida.internos["fase_t"]
    DIS = salida.internos["disparo_t"]
    Phi = salida.internos["base"]

    frecuencias = np.array([[float((S[r] == f).mean()) for f in range(4)]
                            for r in range(cfg.R)]).mean(axis=0)

    dur = {FASES[f]: np.concatenate([_duraciones(S[r], f) for r in range(cfg.R)])
           for f in (SUBIDA, MESETA, BAJADA)}

    pred = np.array([_diagnostico_predictivo(C[r], S[r], cfg.m_diagnostico,
                                             3, cfg.prop_train_diag)
                     for r in range(cfg.R)])
    mez = np.array([_diagnostico_mezcla(C[r], S[r], cfg.m_diagnostico,
                                        3, cfg.prop_train_diag)
                    for r in range(cfg.R)])

    # El salto medido donde se ve: sobre la curva, contra su propia desviacion.
    salto_curva = salida.internos["delta_mu"] @ Phi
    sd_curva = float(salida.curvas.std())

    base.update({
        "d": int(cfg.d),
        "base": cfg.base,
        "max_desvio_ortonormalidad":
            float(salida.internos["info_base"]["max_desvio_ortonormalidad"]),
        "mecanismo": "composicion_estados_4_fases",
        "distribucion_duracion": "poisson_desplazada",
        "perfil_bajada": "exponencial_reescalada",
        "phi": float(cfg.phi),
        "salto": float(cfg.salto),
        "razon_sigma": float(cfg.razon_sigma),
        "c_in": float(cfg.c_in),
        "sd_z_cerrada": float(salida.internos["sd_z"]),
        "frecuencias_fase": {FASES[f]: float(frecuencias[f]) for f in range(4)},
        "frecuencia_fase_min": float(frecuencias.min()),
        "duracion_media": {k: (float(v.mean()) if v.size else float("nan"))
                           for k, v in dur.items()},
        "duracion_media_nominal": {
            "subida": float(cfg.d_min + cfg.lambda_subida),
            "meseta": float(cfg.d_min + cfg.lambda_meseta),
            "bajada": float(cfg.d_min + cfg.lambda_bajada)},
        "n_episodios": int(DIS.sum()),
        "n_episodios_por_replica": float(DIS.sum() / cfg.R),
        "espera_media_en_base": (float(np.mean(_duraciones(S[0], BASE)))
                                 if (S[0] == BASE).any() else float("nan")),
        "amplitud_episodio_curva": float(np.abs(salto_curva).max() / sd_curva),
        "amplitud_episodio_vs_ruido": float(np.abs(salto_curva).max()
                                            / max(cfg.sigma_obs, 1e-12)),
        "m_diagnostico": int(cfg.m_diagnostico),
        "r2_lineal_en_scores": float(np.nanmean(pred[:, 0])),
        "r2_por_fase_en_scores": float(np.nanmean(pred[:, 1])),
        "ganancia_por_fase": float(np.nanmean(pred[:, 1] - pred[:, 0])),
        "persistencia_fase": float(np.mean([
            np.mean([(S[r][1:][S[r][:-1] == k] == k).mean()
                     for k in np.unique(S[r])]) for r in range(cfg.R)])),
        "accuracy_fase_desde_pasado": float(np.nanmean(mez[:, 0])),
        "accuracy_fase_base": float(np.nanmean(mez[:, 1])),
        "inflacion_varianza_mezcla_media": float(np.nanmean(mez[:, 2])),
        "inflacion_varianza_mezcla_p90": float(np.nanmean(mez[:, 3])),
        "razon_var_condicional_entre_fases": float(np.nanmean(mez[:, 4])),
    })
    return base
