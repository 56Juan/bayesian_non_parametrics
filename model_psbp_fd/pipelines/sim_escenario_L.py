"""
sim_escenario_L.py
===================
Escenario L: MEZCLA DE REGRESIONES sobre los COEFICIENTES de una base
ortonormal fija. Escenario de diagnostico, no es un Algoritmo del anexo; se
nombra con la siguiente letra libre despues de J y K, saltando la M porque esa
letra ya designa el numero de componentes FPCA retenidas en todo el proyecto.

Un unico modulo cubre las corridas 41 a 44: son la misma ecuacion generadora
con `modo_conmutacion` en "nivel" o "pendiente" y `n_lags` en 1 o 3. Que el
contraste entre corridas sea exactamente un cambio de parametros, y no de
codigo, es deliberado: cualquier diferencia observada en las metricas no puede
atribuirse a otra cosa.

    corrida 41 : nivel,     n_lags 1
    corrida 42 : nivel,     n_lags 3
    corrida 43 : pendiente, n_lags 1
    corrida 44 : pendiente, n_lags 3

Modelo generador
-----------------
El estado es el vector de coeficientes, NO la curva en la grilla:

    c_t = sum_{(l, J, B) in R_{s_t}}  B c_{t-l, J}  +  eps_t,

    eps_t ~ N(0, diag(sigma_c)^2),   sigma_c_j = decaimiento^j,

    z_t   = <w, c_{t-1}> / sd(z),

    s_t   = 1  si z_t < -c0
            2  si |z_t| <= c0
            3  si z_t > +c0,

y solo al final se sale a la grilla una unica vez:

    X_t(tau) = mu(tau) + sum_j c_{t,j} phi_j(tau).

Cada termino `(l, J, B)` significa: en el regimen s, el BLOQUE de coordenadas
`J` del rezago `l` escribe sobre todas las coordenadas del presente a traves de
`B` (d x |J|). La matriz de regresion del regimen,

    A_s^{(l)} = B en las columnas J, cero en el resto,

tiene entonces soporte de COLUMNAS distinto en cada regimen: el conjunto de
predictores activos conmuta. Eso es "un coeficiente --o un bloque de
coeficientes-- especifica a los otros, despues otro bloque, despues dos", y con
`n_lags = 3` el bloque puede ademas venir de un rezago distinto en cada
regimen: los tres primeros del rezago 1 en uno, los cuatro ultimos del rezago 3
en otro.

Por que bloques y no una sola coordenada
------------------------------------------
La primera version leia UNA coordenada por termino (`A_s = a e_j^T`, rango 1) y
se midio que no alcanzaba: un escalar no puede explicar la varianza de d
coordenadas alimentadas por d innovaciones independientes, de modo que la
autocorrelacion puntual quedaba en 0.05-0.14 y el oraculo apenas recuperaba
0.07-0.32 de la varianza. Con bloques, `A_s` es de rango alto y cada coordenada
del presente recibe una combinacion de varias del pasado.

Leer por bloques ademas ata el escenario a un mecanismo del modelo que ya esta
instrumentado. El PSBPM-FD tiene inclusion por covariable y el pipeline reporta
su PIP por componente y por covariable; un generador en el que el CONJUNTO de
covariables activas conmuta con el regimen es exactamente lo que ese mecanismo
existe para capturar, de modo que la evidencia deja de ser solo "predice mejor"
y pasa a ser "recupera que predictores mandan en cada regimen". Y para los
competidores es mas duro: el FAR estima un unico operador denso y no puede
apagar un bloque segun el estado, mientras que RF y GBT tienen que partir
primero por la variable de regimen y aprender un juego distinto de cortes en
cada rama.

Lo que NO produce la brecha es que los bloques de lectura sean disjuntos, por
mas que parezca lo contrario. Se midio en el espacio de scores truncado a M = 3
con norma_cargas = 0.9: con bloques disjuntos el lineal alcanza 0.402 contra un
techo de 0.450, una ganancia de 0.048 que no da para nada. Si cada bloque, cada
vez que esta activo, empuja en la misma direccion, el predictor lineal lo
incluye con su peso promedio y el danyo es moderado. Ver `construir_regimenes`
para lo que si la produce.

Por que el generador vive en coeficientes y no en la grilla
-------------------------------------------------------------
Los seis Algoritmos del anexo y los escenarios B, T, J y K iteran una
recursion en R^L con un operador integral discretizado y una innovacion
gaussiana correlacionada por un nucleo. Aqui no hay operador de L x L ni
factorizacion de un nucleo en la grilla: la dinamica entera ocurre en R^d con
d = 10, y `Phi` (d x L) aparece una sola vez, al evaluar la trayectoria ya
construida. Simular coeficientes o simular la grilla es la misma cosa
parametrizada distinto --la curva verdadera X_t(tau) queda igual de bien
definida y se evalua en la misma grilla de L puntos--, pero parametrizar en
coeficientes es lo que permite escribir la mezcla de regresiones de forma
exacta y controlar en que direccion del espectro vive la conmutacion.

Consecuencia que hay que declarar al reportar: como la base generadora es
tambien la base de representacion del pipeline, el error de representacion es
exactamente cero y el piso de error de esta corrida no es el de la 30. Es la
misma nota que ya aplica entre la 30 y las 31-34, cuyas bases las elige el GCV
por separado, no una salvedad nueva.

Por que una base ORTONORMAL
----------------------------
Con `W = I` la matriz de Gram desaparece de todas partes a la vez: el FPCA
generalizado `C u = lambda W u` colapsa a un PCA euclideo sobre los
coeficientes, y el blanqueo del FAR con la Cholesky de la Gram
(`theta_w = THETA @ L`) se vuelve la identidad. Ninguno de los dos necesita
adaptacion; el paso simplemente se trivializa. Esa es tambien la razon por la
que el diagnostico puede calcular la rotacion FPCA aqui dentro, con los
autovectores de cov(c), sin importar `FPCA_L2`.

Dos bases cumplen con eso sobre la grilla regular de L puntos, medido con la
cuadratura trapezoidal del proyecto:

    "bspline_lowdin" : B-spline de orden 4 ortonormalizada por Lowdin
                       (phi~ = W^{-1/2} Phi).  max|W - I| ~ 3e-15, d libre.
    "fourier"        : la FourierBasis de skfda ya es ortonormal exacta
                       (max|W - I| ~ 3e-15, cond(W) = 1), pero exige d impar.

Se ortonormaliza por LOWDIN y no por Gram-Schmidt o Cholesky porque W^{-1/2}
es la ortonormalizacion que minimiza la distancia a la base original y por lo
tanto preserva la localizacion: el soporte efectivo de las funciones pasa de
0.09-0.40 del dominio a 0.11-0.57, mientras que Cholesky lo destruye
acumulando soporte hacia un extremo. Con base localizada, "la coordenada j
escribe sobre la carga a" tiene lectura espacial --una region del dominio
gobierna otra en el periodo siguiente-- que con Fourier se pierde.

Haar y Daubechies quedaron descartadas: los saltos diadicos de Haar no caen en
nodos de la grilla de L = 100 y la Gram se desvia 3e-2 de la identidad, que no
es ruido numerico sino sesgo de cuadratura; Daubechies exige PyWavelets, que
no es dependencia del proyecto.

Estacionariedad
----------------
La condicion que se impone no es `rho(A_s) < 1` para cada regimen: con
conmutacion eso no basta (es el problema del radio espectral conjunto, donde
matrices individualmente estables pueden producir un producto explosivo). Se
usa la condicion suficiente por normas

    sum_l ||A_s^{(l)}||_2 <= rho_max < 1   para TODO regimen s,

que da ||c_t|| <= rho_max * max(||c_{t-1}||, ..., ||c_{t-p}||) + ||eps_t|| y
por lo tanto acota la trayectoria con independencia de la secuencia de
regimenes. Con bloques, `A_s^{(l)}` ya no es de rango 1 y su norma espectral no
es la de un vector de carga: se ensambla la matriz d x d de cada rezago --
sumando los terminos que comparten rezago, que no es lo mismo que sumar sus
normas-- y se toma su mayor valor singular. El regimen entero se reescala
despues para que esa suma valga exactamente `norma_cargas`, el parametro que el
usuario fija. No hay calibracion por prueba y error: se normaliza por
construccion y se verifica con assert.

Que separa las corridas: QUE conmuta
--------------------------------------
El par 41/42 conmuta el NIVEL: la pendiente es comun a los tres regimenes y lo
que salta es el intercepto. Esa es la forma funcional exacta de un arbol --un
corte en la variable de umbral y una constante por hoja-- de modo que RF y GBT
no la aproximan sino que la aciertan, mientras el FAR, con un unico intercepto,
no puede representarla. Es el control: un escenario donde ganarle al FAR es
facil y ganarle a los arboles casi imposible.

El par 43/44 conmuta la PENDIENTE: los interceptos son nulos y lo que cambia es
la matriz que multiplica al pasado. El FAR promedia las matrices; los arboles,
que predicen una constante por hoja, tienen que escalonar una recta empinada
DESPUES de haber gastado cortes en la frontera. Ninguno de los dos alcanza el
techo.

Medido en el espacio de scores con M = 3, T0 = 560, una realizacion:

    corrida   modo         FAR     RF   techo   techo-FAR   techo-RF
      41      nivel      0.261  0.411   0.452      0.191      0.041
      42      nivel      0.270  0.403   0.436      0.166      0.033
      43      pendiente  0.205  0.275   0.352      0.147      0.077
      44      pendiente  0.159  0.210   0.271      0.112      0.061

`norma_cargas` NO separa corridas: es el nivel de senyal y vale 0.90 en las
cuatro. Con 0.50 no hay nada que repartir --ningun modelo pasa de 0.07 de R^2--
y la comparacion deja de tener sentido, de modo que no es un punto util del
barrido sino simplemente un escenario sin senyal.

La frontera OBLICUA se probo y se descarto. La idea era castigar a los arboles
haciendo que la frontera no fuera perpendicular a ningun eje, y por si sola es
correcta; pero cuando la conmutacion deja de estar alineada con la coordenada
que las cargas multiplican, la no linealidad se promedia y el techo cae mas de
lo que cae el arbol. Medido con modo "pendiente" y n_lags 1: alineada deja
0.147 de margen sobre el FAR y 0.077 sobre RF, oblicua solo 0.063 y 0.061.
Castiga mas al PSBPM-FD que a sus competidores. Por eso `w` queda alineada al
primer eje en las cuatro corridas y el parametro se conserva solo para poder
reproducir esa medicion.

La calibracion de la escala de z y por que es un punto fijo
------------------------------------------------------------
Los umbrales +-c0 se expresan en unidades de sd(z), de modo que
c0 = 0.4307 = Phi^{-1}(2/3) reparte los tres regimenes en tercios bajo una ley
aproximadamente gaussiana de z. Pero sd(z) depende de la dinamica, que depende
del regimen, que depende de sd(z): es un punto fijo. `_calibrar_escala_z`
lo resuelve iterando unas pocas pasadas de calentamiento --misma idea que la
`saturacion` del Escenario J, medida en un piloto-- y fija el valor resultante
antes de generar la serie que se retiene. Las frecuencias efectivas se
reportan en el diagnostico y no se dan por buenas.

La rotacion FPCA, que es la trampa de este escenario
------------------------------------------------------
La frontera se define en el espacio de coeficientes `c`, pero los modelos no
ven `c`: ven scores FPCA estandarizados. La covarianza estacionaria

    Sigma_c = sum_s p_s (Sigma_c)_{j_s j_s} a^{(s)} a^{(s)T} + Sigma_eps

es una suma de terminos de rango 1 en las direcciones de carga y NO es
diagonal en general, de modo que la FPCA rota y una frontera alineada al eje
en `c` puede llegar oblicua al espacio de scores. Para el PSBPM-FD da igual
--z sigue siendo una combinacion lineal de los scores y el gating probit la
representa exacto--, pero para RF y GBT no: si la rotacion es grande, la
corrida 41 deja de ser el control donde el arbol resuelve la frontera con un
solo corte y se convierte en una version debil de la 43, difuminando el
contraste que las dos corridas existen para medir.

Por eso `resumen_escenario_L` reporta `angulo_w_vs_eje_fpca_grados`: el angulo
entre `w` y el eje principal mas cercano de cov(c). Cerca de 0 grados la
frontera llega alineada y la corrida 41 es un control honesto; cerca de 45 es
oblicua. Es una cantidad que se mide en el piloto y de la que depende la
eleccion de `w`, no un supuesto.

Author: model_psbp_fd
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
from ..utils.quadrature import pesos_trapezoidales

__all__ = [
    "ConfigEscenarioL",
    "base_ortonormal",
    "construir_regimenes",
    "generar_escenario_L",
    "resumen_escenario_L",
    "simular_coeficientes",
]

# Umbral que reparte los tres regimenes en tercios: Phi(0.4307) = 2/3.
C0_TERCIOS = 0.4307


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioL(ConfigObservacion):
    """
    Parametros del Escenario L.

    Hereda de `ConfigObservacion` el esquema de observacion (L, T, burn_in,
    sigma_obs, R, seed, media_fn, jitter) --identico al de las corridas vivas,
    que por eso siguen siendo comparables-- y agrega los del mecanismo.

    Base generadora
        d            : numero de coeficientes simulados. Con "fourier" debe
                       ser impar.
        base         : "bspline_lowdin" o "fourier".
        orden_bspline: orden de la B-spline previa a la ortonormalizacion.

    Dinamica
        n_lags       : 1 o 3 rezagos.
        modo_conmutacion : QUE conmuta con el regimen, y es lo que separa las
                       corridas 41/42 de las 43/44.
                       "nivel"     : la pendiente es comun a los tres
                                     regimenes y lo que salta es el intercepto.
                                     Es la forma funcional exacta de un arbol
                                     --un corte y una constante por hoja-- de
                                     modo que RF y GBT la aciertan y el FAR,
                                     con un unico intercepto, no puede.
                       "pendiente" : el intercepto es nulo en los tres y lo que
                                     conmuta es la matriz que multiplica al
                                     pasado. Ni el FAR ni los arboles alcanzan
                                     el techo: el primero promedia las
                                     matrices, los segundos tienen que
                                     escalonar una recta empinada.
        norma_cargas : sum_l ||A_s|| por regimen. Debe ser < 1 por
                       estacionariedad; con 0.50 no hay senyal que repartir y
                       ningun modelo despega, de modo que 0.90 es el valor util.
        salto_nivel  : magnitud del salto de intercepto, en unidades de la
                       innovacion. Solo interviene con modo "nivel".
        peso_rezago_lejano : fraccion de la norma de cada regimen que se lleva
                       el rezago 3. Solo interviene con `n_lags = 3`; con 1 no
                       hay rezago lejano y el parametro es inerte. NO subirlo
                       sin leer antes la nota de `construir_regimenes`: no
                       compra lo que parece y degrada las dos corridas.
        decaimiento  : sigma_c_j = decaimiento^j. Da el espectro decreciente
                       sin el cual el barrido en M no significaria nada.
        ancho_carga  : ancho del perfil gaussiano de las cargas sobre el
                       indice de coordenada. Con base localizada controla
                       cuanta region del dominio escribe cada regimen.

    Conmutacion
        w            : direccion del indice de conmutacion. None usa e_0
                       (alineada al eje, corridas 41/42). Un vector con varias
                       entradas no nulas la hace oblicua (43/44).
        c0           : umbral en unidades de sd(z).
        iter_calibra : pasadas del punto fijo que calibra sd(z).
    """

    d: int = 10
    base: str = "bspline_lowdin"
    orden_bspline: int = 4

    n_lags: int = 1
    modo_conmutacion: str = "pendiente"
    norma_cargas: float = 0.90
    salto_nivel: float = 1.0
    peso_rezago_lejano: float = 0.20
    decaimiento: float = 0.45
    ancho_carga: float = 1.5

    w: Optional[np.ndarray] = None
    c0: float = C0_TERCIOS
    iter_calibra: int = 3

    # Solo para el diagnostico: en que M se mide la ganancia y donde se corta.
    # No intervienen en la generacion; replican las condiciones del pipeline
    # para que el diagnostico diga si el escenario discrimina DONDE importa.
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
                f"base='fourier' exige d impar; se recibio d={self.d}. "
                "Use d impar o cambie a 'bspline_lowdin', que admite cualquier d."
            )
        if self.n_lags not in {1, 3}:
            raise ValueError(
                f"n_lags={self.n_lags}: este escenario define regimenes para "
                "n_lags 1 o 3."
            )
        if self.modo_conmutacion not in {"nivel", "pendiente"}:
            raise ValueError(
                f"modo_conmutacion={self.modo_conmutacion!r}: debe ser "
                "'nivel' o 'pendiente'."
            )
        if self.modo_conmutacion == "nivel" and self.salto_nivel <= 0:
            raise ValueError("salto_nivel debe ser positivo con modo 'nivel'.")
        if not (0.0 < self.peso_rezago_lejano < 1.0):
            raise ValueError(
                f"peso_rezago_lejano={self.peso_rezago_lejano}: debe estar en "
                "(0, 1); es una fraccion de la norma del regimen."
            )
        if not (0.0 < self.norma_cargas < 1.0):
            raise ValueError(
                f"norma_cargas={self.norma_cargas}: debe estar en (0, 1). La "
                "condicion suficiente de estacionariedad es que la suma de las "
                "normas de las cargas de cada regimen sea menor que uno."
            )
        if not (0.0 < self.decaimiento <= 1.0):
            raise ValueError("decaimiento debe estar en (0, 1].")
        if self.ancho_carga <= 0:
            raise ValueError("ancho_carga debe ser positivo.")
        if self.c0 <= 0:
            raise ValueError("c0 debe ser positivo.")
        if self.iter_calibra < 1:
            raise ValueError("iter_calibra debe ser al menos 1.")
        if self.w is not None and np.asarray(self.w).shape != (self.d,):
            raise ValueError(
                f"w debe tener forma ({self.d},); se recibio "
                f"{np.asarray(self.w).shape}."
            )
        if self.burn_in < 50:
            raise ValueError(
                "burn_in debe ser al menos 50: la calibracion de sd(z) usa el "
                "calentamiento y con pocas pasadas el punto fijo no converge."
            )


# ==========================================================================
# BASE ORTONORMAL
# ==========================================================================

def base_ortonormal(
    tau: np.ndarray,
    d: int,
    base: str = "bspline_lowdin",
    orden_bspline: int = 4,
) -> Tuple[np.ndarray, dict]:
    """
    Construye `Phi` (d, L) ortonormal en L^2 bajo la cuadratura trapezoidal.

    Retorna la base y un diccionario con la desviacion medida respecto de la
    identidad, que el diagnostico publica en vez de darla por supuesta: la
    ortonormalidad es la propiedad de la que dependen el FPCA euclideo y el
    blanqueo trivial del FAR, y conviene que sea verificable desde el
    artefacto.
    """
    try:
        from skfda.representation.basis import BSplineBasis, FourierBasis
    except ImportError as e:
        raise ImportError(
            "skfda es requerido. Instalar con: pip install scikit-fda"
        ) from e

    w_quad = pesos_trapezoidales(tau)

    if base == "fourier":
        basis = FourierBasis(domain_range=(0.0, 1.0), n_basis=d, period=1.0)
        Phi = np.asarray(basis(tau)).reshape(d, -1)
    elif base == "bspline_lowdin":
        basis = BSplineBasis(domain_range=(0.0, 1.0), n_basis=d,
                             order=orden_bspline)
        Phi_cruda = np.asarray(basis(tau)).reshape(d, -1)
        W = (Phi_cruda * w_quad) @ Phi_cruda.T
        evals, U = np.linalg.eigh(W)
        if evals.min() <= 0:
            raise np.linalg.LinAlgError(
                f"La Gram de la B-spline (d={d}, orden={orden_bspline}) tiene "
                f"autovalor minimo {evals.min():.3e} <= 0: la base es "
                "numericamente degenerada. Reduzca d o suba el orden."
            )
        Phi = (U @ np.diag(evals ** -0.5) @ U.T) @ Phi_cruda
    else:
        raise ValueError(f"base={base!r} no reconocida.")

    W_final = (Phi * w_quad) @ Phi.T
    info = {
        "base": base,
        "d": int(d),
        "max_desvio_ortonormalidad": float(np.abs(W_final - np.eye(d)).max()),
        "cond_gram": float(np.linalg.cond(W_final)),
    }
    return Phi, info


# ==========================================================================
# REGIMENES
# ==========================================================================

def _perfil_carga(d: int, centro: float, ancho: float) -> np.ndarray:
    """Vector de carga unitario con perfil gaussiano sobre el indice."""
    j = np.arange(d, dtype=float)
    a = np.exp(-0.5 * ((j - centro) / ancho) ** 2)
    return a / np.linalg.norm(a)


def _bloque_carga(d: int, bloque: Sequence[int], centro: float,
                  ancho: float) -> np.ndarray:
    """
    Matriz `B` (d, |bloque|): la k-esima coordenada leida escribe sobre un
    perfil localizado centrado en `centro + k`.

    Que la escritura sea localizada y se desplace con k es lo que conserva la
    lectura espacial de la base: con `bspline_lowdin`, el bloque leido es una
    region del dominio y la escritura otra, desplazandose en paralelo. Con una
    `B` aleatoria el mecanismo seria el mismo pero sin nada que interpretar.
    """
    return np.column_stack([_perfil_carga(d, centro + k, ancho)
                            for k in range(len(bloque))])


def _matrices_por_lag(terminos, d: int) -> dict:
    """
    Ensambla `A_s^{(l)}` (d, d) por rezago, sumando los terminos que comparten
    rezago. Sumar las matrices y no sus normas importa: dos terminos del mismo
    rezago pueden cancelarse parcialmente y la suma de normas sobreestimaria la
    norma efectiva, apretando la condicion de estacionariedad mas de lo debido.
    """
    A = {}
    for lag, bloque, B in terminos:
        M = A.setdefault(lag, np.zeros((d, d)))
        M[:, np.asarray(bloque)] += B
    return A


def construir_regimenes(
    d: int,
    n_lags: int,
    norma_cargas: float,
    ancho_carga: float = 1.5,
    modo_conmutacion: str = "pendiente",
    salto_nivel: float = 1.0,
    peso_rezago_lejano: float = 0.20,
) -> Tuple[List[List[Tuple[int, np.ndarray, np.ndarray]]], np.ndarray]:
    """
    Devuelve `(regimenes, constantes)`: los tres regimenes como listas de
    terminos `(lag, bloque, B)`, y la matriz (3, d) de interceptos.

    Los regimenes 1 y 2 aplican al bloque bajo del rezago 1 la misma carga con
    signo opuesto; el regimen 3 le aplica una carga distinta y ademas activa el
    bloque alto. Con `n_lags = 3` ese bloque alto viene del rezago 3 en vez del
    1, de modo que el regimen elige tambien de que rezago viene la informacion.
    Es la misma estructura en los dos casos y por eso un unico bucle de
    simulacion los cubre.

    Los regimenes 1 y 2 leen EL MISMO bloque con cargas de signo opuesto, y esa
    --y no el cambio de bloque-- es la condicion de la que depende que el
    escenario discrimine. Se midio: lo que un predictor lineal unico no puede
    representar no es que el conjunto de predictores activos cambie, sino que
    el MISMO predictor tenga efectos opuestos segun el estado. Si cada bloque,
    cuando esta activo, siempre empuja en la misma direccion, el lineal lo
    incluye con su peso promedio y el danyo es moderado. Sobre la misma
    realizacion con norma_cargas = 0.9, medido en el espacio de scores
    truncado a M = 3, que es donde vive el modelo:

        bloques disjuntos          lineal 0.402, techo 0.450, ganancia 0.048
        mismo bloque, cargas +-B   lineal 0.294, techo 0.475, ganancia 0.182
        mismo bloque, ortogonales  lineal 0.346, techo 0.445, ganancia 0.098

    El bloque ALTO entra solo en el regimen 3, de modo que el conjunto de
    covariables activas sigue conmutando y el PIP --que el pipeline ya reporta
    por componente y por covariable-- tiene algo que recuperar; pero es la capa
    secundaria, no el mecanismo que produce la brecha.

    La simetria +-B no es exacta a proposito. Con los tres regimenes
    perfectamente antisimetricos la respuesta se vuelve par, el predictor
    lineal degenera casi a la media incondicional y el resultado es facil de
    atacar como construido para ganar; el regimen 3, con una carga distinta y
    un bloque adicional, rompe esa simetria y deja al FAR con senyal real que
    recuperar (0.294 de R^2, no cero).

    Todos los bloques escriben sobre coordenadas BAJAS (`centro` pequenyo) por
    dos razones que se midieron y no se supusieron:

    1.  REALIMENTACION. El regimen se decide con `<w, c_{t-1}>`, de modo que si
        las cargas no escriben sobre las coordenadas que se leen, esas
        coordenadas son ruido blanco puro: el indice z_t queda iid, los
        regimenes no persisten y no hay "un cluster induce al siguiente".

    2.  VISIBILIDAD BAJO TRUNCAMIENTO. `sigma_c` decae con el indice, asi que
        las coordenadas altas son las componentes FPCA de varianza pequenya.
        Escribir la senyal ahi la pone justo donde el barrido en M (1, 2, 3) no
        la ve: el modelo observa las de mayor varianza, que serian ruido.

    El regimen entero se reescala para que `sum_l ||A_s^{(l)}||_2` valga
    exactamente `norma_cargas`, que es a la vez la condicion suficiente de
    estacionariedad y la pendiente dentro del regimen.
    """
    if n_lags not in (1, 3):
        raise ValueError(f"n_lags={n_lags}: solo 1 o 3.")

    b_bajo = list(range(0, 3))
    b_alto = list(range(max(0, d - 4), d))
    lag_alto = 3 if n_lags == 3 else 1

    B0 = _bloque_carga(d, b_bajo, 0.0, ancho_carga)
    B3 = _bloque_carga(d, b_bajo, 3.0, ancho_carga)
    BA = _bloque_carga(d, b_alto, 0.0, ancho_carga)

    # Termino del rezago lejano. Con n_lags = 3 entra en LOS TRES regimenes y
    # sobre el bloque BAJO --no solo en el tercero y sobre el alto, donde la
    # varianza ya decayo y su contribucion promedio era despreciable--. Es
    # COMUN a los tres, o sea no conmuta: senyal lineal bien especificada que
    # un FAR(3) puede capturar y un FAR(1) no.
    #
    # CUANTO puede pesar esta acotado por la estacionariedad y no por gusto, y
    # es la razon de que `peso_rezago_lejano` sea bajo. Con `norma_cargas` en
    # 0.90 el radio espectral de la companera por regimen ya vale 0.912, y con
    # 1.30 la trayectoria diverge: el presupuesto total de dependencia esta
    # agotado, de modo que repartirlo entre el rezago 1 y el 3 es un juego de
    # suma cero que ademas se lo quita al mecanismo de conmutacion. Medido
    # sobre SEIS replicas, R^2 en scores con M = 3 (error estandar ~0.03):
    #
    #   modo       peso   FAR(1)  FAR(3)     RF   techo   techo-RF
    #   nivel      0.05    0.257   0.255  0.399   0.460      0.061
    #   nivel      0.20    0.191   0.190  0.329   0.427      0.097
    #   nivel      0.35    0.103   0.141  0.274   0.379      0.105
    #   pendiente  0.05    0.189   0.178  0.242   0.286      0.044
    #   pendiente  0.35    0.097   0.103  0.072   0.108      0.036
    #
    # `FAR(3) - FAR(1)` vale -0.002, -0.001 y +0.038: ni con el peso mas alto
    # se separa del ruido, y ese +0.038 cuesta 0.08 de techo y le quita a la
    # corrida 42 su papel de control. Subir este parametro NO hace que el
    # rezago 3 importe; solo diluye la senyal de todos.
    lejano = [(lag_alto, b_bajo, B0)] if n_lags == 3 else []

    if modo_conmutacion == "pendiente":
        crudos = [
            [(1, b_bajo, B0)] + lejano,
            [(1, b_bajo, -B0)] + lejano,
            [(1, b_bajo, B3), (lag_alto, b_alto, BA)],
        ]
        constantes = np.zeros((3, d))
    elif modo_conmutacion == "nivel":
        # La MISMA pendiente en los tres; lo que salta es el intercepto, con
        # la misma estructura antisimetrica que el modo "pendiente" aplica a
        # la matriz. Las constantes suman cero para no desplazar la media
        # marginal de la curva y dejar `mu` como la unica media del proceso.
        crudos = [[(1, b_bajo, B0)] + lejano for _ in range(3)]
        salto = salto_nivel * _perfil_carga(d, 0.0, ancho_carga)
        constantes = np.vstack([salto, -salto, np.zeros(d)])
    else:
        raise ValueError(f"modo_conmutacion={modo_conmutacion!r}: "
                         "solo 'nivel' o 'pendiente'.")

    regimenes = []
    for terminos in crudos:
        # Reparto entre rezago 1 y rezago lejano ANTES de normalizar: sin el,
        # el peso relativo del rezago lejano seria un accidente de cuantos
        # terminos tiene cada regimen. Con n_lags = 1 no hay rezago lejano y
        # este bloque no hace nada.
        cerca = [t for t in terminos if t[0] == 1]
        lejos = [t for t in terminos if t[0] != 1]
        if lejos:
            n_c = sum(np.linalg.norm(M, 2)
                      for M in _matrices_por_lag(cerca, d).values())
            n_l = sum(np.linalg.norm(M, 2)
                      for M in _matrices_por_lag(lejos, d).values())
            terminos = ([(l, b, B * (1.0 - peso_rezago_lejano) / n_c)
                         for l, b, B in cerca]
                        + [(l, b, B * peso_rezago_lejano / n_l)
                           for l, b, B in lejos])

        A = _matrices_por_lag(terminos, d)
        suma = sum(float(np.linalg.norm(M, 2)) for M in A.values())
        escala = norma_cargas / suma
        regimenes.append([(l, np.asarray(bloque), B * escala)
                          for l, bloque, B in terminos])
    return regimenes, constantes


def _norma_por_regimen(regimenes, d: int) -> np.ndarray:
    """sum_l ||A_s^{(l)}||_2 por regimen, con la matriz de cada rezago armada."""
    return np.array([
        sum(float(np.linalg.norm(M, 2))
            for M in _matrices_por_lag(terminos, d).values())
        for terminos in regimenes
    ])


# ==========================================================================
# DINAMICA DEL ESCENARIO
# ==========================================================================

def _paso(regimenes, constantes, s: int,
          historia: Sequence[np.ndarray]) -> np.ndarray:
    """
    Aplica el regimen `s`. `historia[l - 1]` es c_{t-l}.

    Cada termino lee el BLOQUE de coordenadas que le toca del rezago que le
    toca y lo escribe sobre todas las del presente a traves de su matriz; al
    resultado se suma el intercepto del regimen, nulo salvo en modo "nivel".
    """
    out = np.array(constantes[s], dtype=float)
    for lag, bloque, B in regimenes[s]:
        out += B @ historia[lag - 1][bloque]
    return out


def _regimen(z: float, c0: float) -> int:
    """Tres regimenes por dos umbrales simetricos sobre el indice z."""
    if z < -c0:
        return 0
    if z > c0:
        return 2
    return 1


def _calibrar_escala_z(
    regimenes,
    constantes: np.ndarray,
    w: np.ndarray,
    sigma_c: np.ndarray,
    c0: float,
    n_lags: int,
    burn_in: int,
    iter_calibra: int,
    rng: np.random.Generator,
) -> float:
    """
    Resuelve el punto fijo de sd(z): los umbrales se expresan en unidades de
    sd(z), pero sd(z) depende del regimen, que depende de los umbrales.

    Se itera unas pocas pasadas de calentamiento partiendo de sd(z) = 1 y
    quedandose con la desviacion empirica de la pasada anterior. Converge en
    dos o tres iteraciones porque la dependencia es debil: el regimen afecta a
    z solo a traves de la varianza estacionaria, no de su escala directa.
    """
    sd_z = 1.0
    for _ in range(iter_calibra):
        historia = [rng.normal(0.0, sigma_c) for _ in range(n_lags)]
        zs = np.empty(burn_in)
        for t in range(burn_in):
            bruto = float(w @ historia[0])
            zs[t] = bruto
            c = _paso(regimenes, constantes, _regimen(bruto / sd_z, c0), historia)
            c += rng.normal(0.0, sigma_c)
            historia = [c] + historia[:-1]
        nueva = float(zs.std(ddof=1))
        if not np.isfinite(nueva) or nueva <= 0:
            raise FloatingPointError(
                "La calibracion de sd(z) produjo un valor no positivo o no "
                "finito: la trayectoria diverge. Revise norma_cargas."
            )
        sd_z = nueva
    return sd_z


def simular_coeficientes(
    regimenes,
    constantes: np.ndarray,
    w: np.ndarray,
    sigma_c: np.ndarray,
    sd_z: float,
    c0: float,
    n_lags: int,
    T: int,
    burn_in: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Itera la mezcla de regresiones en R^d y devuelve `(C, S)` con C de forma
    (T, d) los coeficientes retenidos y S de forma (T,) el regimen de cada
    periodo, base 0.

    La grilla no interviene en ningun punto de esta funcion: el estado es el
    vector de coeficientes y nada mas. El regimen de cada periodo se decide
    siempre con el rezago 1, con independencia de que rezagos use el regimen
    para escribir, porque la frontera es funcion del estado mas reciente.
    """
    d = sigma_c.size
    historia = [rng.normal(0.0, sigma_c) for _ in range(n_lags)]

    for _ in range(burn_in):
        z = float(w @ historia[0]) / sd_z
        c = _paso(regimenes, constantes, _regimen(z, c0), historia)
        c += rng.normal(0.0, sigma_c)
        historia = [c] + historia[:-1]

    C = np.empty((T, d))
    S = np.empty(T, dtype=int)
    for t in range(T):
        z = float(w @ historia[0]) / sd_z
        s = _regimen(z, c0)
        c = _paso(regimenes, constantes, s, historia)
        c += rng.normal(0.0, sigma_c)
        historia = [c] + historia[:-1]
        C[t] = c
        S[t] = s

    if not np.all(np.isfinite(C)):
        raise FloatingPointError(
            "La trayectoria de coeficientes diverge. La condicion suficiente "
            "de estacionariedad es sum_l ||a|| < 1 por regimen; revise "
            "norma_cargas."
        )
    return C, S


# ==========================================================================
# GENERADOR PRINCIPAL
# ==========================================================================

def generar_escenario_L(cfg: ConfigEscenarioL) -> SalidaSimulacion:
    """
    Genera R replicas independientes del Escenario L.

    La base, los regimenes y la escala de la innovacion no dependen de la
    realizacion y se construyen una sola vez. La calibracion de sd(z) usa un
    generador propio derivado de la semilla maestra, de modo que el valor
    calibrado es el mismo para todas las replicas y no introduce una diferencia
    entre ellas que se confundiria con variabilidad Monte Carlo.
    """
    cfg.validar()

    tau = grilla_regular(cfg.L)
    mu = evaluar_media(cfg.media_fn, tau)
    Phi, info_base = base_ortonormal(tau, cfg.d, cfg.base, cfg.orden_bspline)

    regimenes, constantes = construir_regimenes(
        cfg.d, cfg.n_lags, cfg.norma_cargas, cfg.ancho_carga,
        cfg.modo_conmutacion, cfg.salto_nivel, cfg.peso_rezago_lejano)
    normas = _norma_por_regimen(regimenes, cfg.d)
    assert normas.max() < 1.0, (
        f"sum_l ||A_s|| = {normas.max():.4f} >= 1 en algun regimen: no se "
        "cumple la condicion suficiente de estacionariedad."
    )

    sigma_c = cfg.decaimiento ** np.arange(cfg.d, dtype=float)
    if cfg.w is None:
        w = np.zeros(cfg.d)
        w[0] = 1.0
    else:
        w = np.asarray(cfg.w, dtype=float).copy()
    w = w / np.linalg.norm(w)

    rng_cal = np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(1)[0])
    sd_z = _calibrar_escala_z(regimenes, constantes, w, sigma_c, cfg.c0,
                              cfg.n_lags, cfg.burn_in, cfg.iter_calibra, rng_cal)

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)

    curvas = np.empty((cfg.R, cfg.T, cfg.L))
    observaciones = np.empty((cfg.R, cfg.T, cfg.L))
    coeficientes = np.empty((cfg.R, cfg.T, cfg.d))
    regimen_t = np.empty((cfg.R, cfg.T), dtype=int)

    for r, semilla in enumerate(hijas):
        rng = np.random.default_rng(semilla)
        C, S = simular_coeficientes(regimenes, constantes, w, sigma_c, sd_z,
                                    cfg.c0, cfg.n_lags, cfg.T, cfg.burn_in, rng)
        # Unico punto donde interviene la grilla.
        curvas_r = mu[None, :] + C @ Phi
        curvas[r] = curvas_r
        observaciones[r] = aplicar_ruido_observacion(curvas_r, cfg.sigma_obs, rng)
        coeficientes[r] = C
        regimen_t[r] = S

    salida = SalidaSimulacion(
        observaciones=observaciones,
        curvas=curvas,
        grilla=tau,
        media=mu,
        semillas=registro,
        config=cfg,
        internos={
            "base": Phi,
            "info_base": info_base,
            "regimenes": regimenes,
            "constantes": constantes,
            "coeficientes": coeficientes,
            "regimen_t": regimen_t,
            "w": w,
            "sd_z": sd_z,
            "sigma_c": sigma_c,
            "normas_regimen": normas,
            "pesos_cuadratura": pesos_trapezoidales(tau),
        },
    )
    salida.diagnostico = resumen_escenario_L(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD ESPECIFICO
# ==========================================================================

def _diagnostico_predictivo(C: np.ndarray, S: np.ndarray, M: int,
                            n_lags: int, prop_train: float) -> Tuple[float, float]:
    """
    R^2 fuera de muestra del mejor predictor LINEAL y del techo alcanzable
    conociendo el regimen, ambos EN EL ESPACIO DE SCORES truncado a `M`.

    Se mide sobre los scores y no sobre los coeficientes porque es ahi donde
    vive el modelo, y las dos cosas no coinciden. Con base ortonormal la FPCA
    en metrica L^2 es el PCA euclideo de `cov(C)`, de modo que proyectar sobre
    sus `M` primeros autovectores reproduce exactamente las covariables que
    veran el FAR, RF, GBT y el PSBPM-FD.

    El techo es una regresion POR REGIMEN estimada con el regimen conocido, no
    la media condicional exacta del generador. La diferencia importa y se
    midio: el oraculo exacto en R^d daba brechas de 0.08-0.10 alli donde lo
    realmente alcanzable por un modelo que ve M componentes y tiene que estimar
    era 0.02-0.05. Reportar el oraculo exacto sobreestima lo que el PSBPM-FD
    puede ganar y llevaria a dar por bueno un escenario que no discrimina.

    La diferencia entre ambos es la cifra que decide si el escenario sirve. Ha
    de ser sustantiva, y el R^2 lineal NO ha de ser nulo: si la referencia
    lineal degenera a la media incondicional el resultado es facil de atacar
    como construido para ganar --la misma advertencia que el Escenario J
    documenta para sus potencias impares--.
    """
    M = int(min(M, C.shape[1]))
    U = np.linalg.eigh(np.cov(C.T))[1][:, ::-1][:, :M]
    Z = C @ U

    Y = Z[n_lags:]
    X = np.hstack([np.ones((Y.shape[0], 1))]
                  + [Z[n_lags - l: Z.shape[0] - l] for l in range(1, n_lags + 1)])
    s = S[n_lags:]

    corte = int(prop_train * C.shape[0]) - n_lags
    tr, te = np.arange(corte), np.arange(corte, Y.shape[0])
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
    r2_reg = float(1.0 - ((Y[te] - P[te]) ** 2).sum() / sst)
    return r2_lin, r2_reg


def _angulo_w_vs_eje_fpca(C: np.ndarray, w: np.ndarray) -> float:
    """
    Angulo en grados entre `w` y el eje principal mas cercano de cov(C).

    Con base ortonormal la FPCA en metrica L^2 es un PCA euclideo sobre los
    coeficientes, de modo que estos autovectores SON las direcciones de score
    que veran los modelos. Cerca de 0 grados la frontera llega alineada al eje
    y la corrida con `w = e_0` es un control honesto donde un arbol la resuelve
    con un solo corte; cerca de 45 grados llega oblicua y el contraste entre
    las dos corridas se difumina.
    """
    _, U = np.linalg.eigh(np.cov(C.T))
    cos_max = float(np.abs(U.T @ w).max())
    return float(np.degrees(np.arccos(np.clip(cos_max, 0.0, 1.0))))


def resumen_escenario_L(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad del generador, previo a cualquier ajuste.

    Extiende `diagnostico_comun` con lo propio de la mezcla de regresiones: la
    condicion de estacionariedad efectivamente impuesta, el reparto empirico de
    los regimenes contra el tercio nominal, la desviacion de la base respecto
    de la ortonormalidad, cuanto recupera el mejor predictor lineal, y el
    angulo con que la frontera llega al espacio de scores.

    Los cuatro ultimos son los que deciden si el escenario sirve: un reparto
    muy desbalanceado deja un regimen sin datos para estimar, un R^2 lineal
    alto significa que el FAR recupera la conmutacion y el escenario no
    discrimina, y un angulo grande en la corrida de control la convierte en una
    version debil de la corrida oblicua.
    """
    if not isinstance(salida.config, ConfigEscenarioL):
        raise TypeError(
            "resumen_escenario_L requiere una salida generada con "
            f"ConfigEscenarioL; se recibio {type(salida.config).__name__}."
        )

    base = diagnostico_comun(salida)
    cfg = salida.config
    C = salida.internos["coeficientes"]
    S = salida.internos["regimen_t"]
    w = salida.internos["w"]

    frecuencias = np.array([[float((S[r] == s).mean()) for s in range(3)]
                            for r in range(cfg.R)])
    pred = np.array([_diagnostico_predictivo(C[r], S[r], cfg.m_diagnostico,
                                             cfg.n_lags, cfg.prop_train_diag)
                     for r in range(cfg.R)])
    r2, r2_reg = pred[:, 0], pred[:, 1]
    angulo = np.array([_angulo_w_vs_eje_fpca(C[r], w) for r in range(cfg.R)])

    Sigma_c = np.cov(C[0].T)
    fuera = Sigma_c - np.diag(np.diag(Sigma_c))

    base.update({
        "d": int(cfg.d),
        "base": cfg.base,
        "max_desvio_ortonormalidad":
            float(salida.internos["info_base"]["max_desvio_ortonormalidad"]),
        "n_lags": int(cfg.n_lags),
        "modo_conmutacion": cfg.modo_conmutacion,
        "norma_cargas": float(cfg.norma_cargas),
        "peso_rezago_lejano": float(cfg.peso_rezago_lejano),
        "norma_max_regimen": float(salida.internos["normas_regimen"].max()),
        "estacionariedad_ok": bool(salida.internos["normas_regimen"].max() < 1.0),
        "sd_z_calibrada": float(salida.internos["sd_z"]),
        "frecuencias_regimen": frecuencias.mean(axis=0).tolist(),
        "frecuencia_regimen_min": float(frecuencias.mean(axis=0).min()),
        "m_diagnostico": int(cfg.m_diagnostico),
        "r2_lineal_en_scores": float(r2.mean()),
        "r2_por_regimen_en_scores": float(r2_reg.mean()),
        "ganancia_por_regimen": float(r2_reg.mean() - r2.mean()),
        "persistencia_regimen": float(np.mean([
            np.mean([(S[r][1:][S[r][:-1] == k] == k).mean() for k in range(3)])
            for r in range(cfg.R)])),
        "angulo_w_vs_eje_fpca_grados": float(angulo.mean()),
        "diagonalidad_sigma_c": float(
            np.abs(fuera).max() / np.abs(np.diag(Sigma_c)).max()
        ),
    })
    return base
