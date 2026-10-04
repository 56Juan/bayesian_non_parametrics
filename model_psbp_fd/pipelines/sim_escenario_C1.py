"""
sim_escenario_C1.py
====================
Algoritmo C-1 del anexo (`docs/01 Anexo.tex`, `ane_00_03_01_alg_c1`): mezcla de
K = 3 mecanismos LINEALES en el espacio de los scores, con asignacion Z_t
dependiente de los scores de los L = 2 rezagos. Aloja ademas el motor comun a
C-1, C-2 y C-3 (`simular_mezcla_scores`) y su control de calidad
(`resumen_mezcla_scores`), como `sim_escenario_B1` aloja el de la seccion B.

No confundir con `sim_escenario_C.py`: ese modulo implementa OTRO diseno
(coeficientes de Fourier con un impulsor AR(p) y una respuesta cuadratica, sin
mecanismos ni Z_t) que no coincide con el anexo vigente y no lo usa ninguna
corrida viva. De el solo se reutiliza `base_fourier`.

Modelo generador (eq:ane_coef_curva, eq:ane_algC1, eq:ane_algC1_asignacion)
-------------------------------------------------------------------------------
    X_t(tau) = mu(tau) + sum_{j=1}^{J} xi_tj phi_j(tau),            J = 10,

    P(Z_t = k | xi_{t-1}, ..., xi_{t-L}) = softmax_k q_k(xi_{t-1}, ..., xi_{t-L}),
    xi_t = mu_{Z_t} + sum_{l=1}^{L} A_{Z_t,l} xi_{t-l} + eps_t,       L = 2.

La asignacion depende del VECTOR de scores en los L rezagos, no de cada
componente por separado. mu(tau) = sin(2 pi tau) (`media_seno`, la misma
definicion de la seccion B) y phi_j son FIJAS y conocidas.

Lo que el anexo deja abierto y como se lee aqui
-------------------------------------------------
1. Ambiguedad de C-1. La ecuacion escribe A_{Z_t,l} (una matriz por mecanismo),
   pero la introduccion de C-2 dice que en C-1 "todos los mecanismos comparten
   la misma estructura de dependencia temporal". Se lee: en C-1 los mecanismos
   comparten el PATRON (los mismos pares (score, rezago) no nulos en A_{k,l}) y
   difieren en los VALORES de mu_k y A_{k,l}; en C-2 difiere el patron. Todas
   las entradas del patron son no nulas en los tres mecanismos (`A_TILDE_C1`).
2. "q_k lineal" se lee AFIN: q_k = c_k + sum_l g_{k,l}' xi_{t-l}, con q_1 = 0
   de referencia. Sin intercepto el mecanismo central nunca pasa de 1/3.
3. phi_j: base de Fourier ortonormal en L^2 sin constante (`base_fourier`),
   phi_{2r-1} = sqrt(2) sin(2 pi r tau), phi_{2r} = sqrt(2) cos(2 pi r tau).
4. Innovacion: eps_t ~ N(0, diag(sigma_j^2)), iid en t, independiente de Z_t y
   comun a los tres mecanismos (el anexo no la especifica).
5. Calentamiento B = 300 periodos, igual que la seccion B (el Cuadro
   tab:ane_observacion no fija B para C). Arranque en xi = 0.

Escala: coordenadas estandarizadas z = xi / s
---------------------------------------------
La dinamica se especifica sobre z_tj = xi_tj / s_j, con s_j = sqrt(lambda_j) y
espectro NOMINAL geometrico lambda_j = lambda_1 rho^(j-1) (lambda_1 = 0.5, rho =
0.55). Es el mismo proceso del anexo con

    mu_k = s * mu~_k,   A_{k,l} = diag(s) A~_{k,l} diag(s)^-1,
    q_k(xi) = q~_k(xi / s),   Var(eps_tj) = s_j^2 sigma~_j^2,

y `internos` guarda las cantidades en escala xi (`medias_mecanismos`,
`matrices_dependencia`). Trabajar en z hace que un coeficiente cruzado se lea
igual cualquiera sea la varianza de las componentes que conecta. El espectro
REALIZADO es lambda_j Var(z_j) y no el nominal, porque la mezcla infla la
varianza del bloque activo (ver cifras abajo); sigue siendo decreciente.

Por que un espectro decreciente y por que rho = 0.55: el anexo llama a xi_tj el
score de la j-esima componente principal, asi que el orden de las phi_j tiene
que ser el de la varianza, o la FPCA estimada no puede alinearse con ellas. La
separacion entre autovalores consecutivos (razon ~2) es lo que protege esa
alineacion de las correlaciones contemporaneas que la dinamica cruzada induce:
el angulo de rotacion del par (j, j+1) es ~ corr sqrt(lambda_j lambda_{j+1}) /
(lambda_j - lambda_{j+1}). Con rho = 0.55 la regla del 95 % cae en M = 5.

Bloque activo y bloque pasivo
-----------------------------
La dinamica que distingue a los mecanismos vive en las cuatro primeras
componentes (`BLOQUE_ACTIVO = 4`). Las componentes 5..10 son AR(1) propios con
coeficiente 0.3 y varianza unitaria en z, IDENTICOS en los tres mecanismos y
que no alimentan al bloque activo: completan J = 10 y pueblan la cola del
espectro. Consecuencia: E[xi_j | pasado] del bloque activo solo necesita los L
rezagos de xi_1..xi_4, de modo que M = 4 es el primer M que ve toda la
dinamica y M > 4 agrega covariables sin informacion.

Diseno de C-1 (coordenadas z; tablas `M_TILDE_C1`, `A_TILDE_C1`, `C_ASIG_C1`,
`G_ASIG_C1`)
---------------------------------------------------------------------------
Tres mecanismos: 1 = centro, 2 = nivel alto, 3 = nivel bajo. mu~_k solo se
separa en xi_1: (0, +1, -1). La asignacion es autoreforzante:

    q_1 = 0,
    q_2 = -2.1 + 1.75 z_{1,t-1} + 0.7 z_{1,t-2} + 0.5 z_{2,t-1},
    q_3 = -2.1 - 1.75 z_{1,t-1} - 0.7 z_{1,t-2} + 0.5 z_{2,t-1},

el nivel alto de xi_1 favorece al mecanismo que lo mantiene alto, y xi_2 alto
empuja a salir del centro hacia cualquiera de los dos extremos. La serie de
xi_1 dibuja fases: meseta alta, meseta baja y tramos alrededor de cero.

Por que el salto es grande y la persistencia propia chica. La permanencia en
un regimen la pone la ASIGNACION, no el AR de xi_1: en los extremos el
coeficiente propio es 0.2 (+0.1 en el rezago 2) y el nivel del regimen es
delta / 0.7 ~ 1.4, contra una sd intra-regimen ~0.46. Asi, dado el pasado, la
ley de xi_1 es una mezcla de dos modos separados (~2.7 sd) con pesos ~0.8/0.2
cerca de un cambio, que es la multimodalidad que el anexo pide. Se probo la
alternativa --salto chico (0.45) y AR propio 0.5 en los extremos--: da
rachas parecidas pero la fraccion de varianza entre mecanismos de xi_1 cae a
0.05 (casi unimodal), y con salto grande Y AR alto los regimenes se vuelven
absorbentes (rachas > 30, el centro por debajo del 7 %).

Patron comun de A~_{k,l} (bloque activo, (respuesta <- covariable)):

    rezago 1: diagonal 1..4,  (2 <- 1),  (3 <- 2)
    rezago 2: (1 <- 1),       (4 <- 3)

y los valores cambian con el mecanismo; los cruzados cambian de SIGNO:

    k   diag rezago 1        (2<-1)  (3<-2)  (1<-1) r2  (4<-3) r2
    1   0.2 0.3 0.3 0.3       0.2     0.8     -0.1       -0.6
    2   0.2 0.4 0.2 0.2       0.5    -0.3      0.1        0.3
    3   0.2 0.4 0.2 0.2      -0.5    -0.3      0.1        0.3

El signo alternado es el diseno: (2<-1) vale +0.5 en el nivel alto y -0.5 en el
bajo, de modo que xi_2 sube en LOS DOS extremos --responde a |xi_1|, no a
xi_1--; un predictor lineal promedia esos coeficientes y ve casi cero, la
mezcla los separa. Los pesos de (3<-2) y (4<-3) se eligieron para que su
promedio ponderado por la ocupacion (y por la energia de cada mecanismo) quede
cerca de cero: asi la correlacion contemporanea del bloque activo se mantiene
baja y la base de Fourier sigue siendo, aproximadamente, la de autofunciones.
sigma~ = (0.45, 0.75, 0.8, 0.8) en el bloque activo y sqrt(0.91) en el pasivo.
Se calibra, en cada algoritmo, para que Var(z_j) ~ 1 en j = 2..4 --el espectro
realizado sigue al nominal-- y xi_1 queda en ~1.2 lambda_1 por la varianza
entre regimenes. Sin esa calibracion, en C-2 y C-3 lambda_4 realizado quedaba
a 1.3x de lambda_5 y la FPCA estimada podia mezclar las componentes 4 y 5.

Cifras de calibracion
---------------------
Tres fuentes: (a) "poblacional", 10 replicas de T = 3000 sin ruido (20 x
10000 para estabilidad); (b) la realizacion de las corridas vivas, T = 1000,
seed 41232; (c) el pipeline del _01 sobre (b) --GCV, FPCA en train, sigma_obs
= 0.25--, medido con un script suelto. R^2 en L^2 = agregado sobre las J
componentes; "lineal" = MCO sobre los L rezagos de las J componentes.

    ocupacion (centro, alto, bajo)   (a) 0.45 0.29 0.26    (b) 0.45 0.27 0.28
    racha media                      (a) 3.5  4.3  4.1     (b) 3.6  3.9  4.8
    transiciones en T = 1000         (b) 253, 74 de ellas en el bloque de prueba
    R^2 L^2 oraculo / lineal         (a) 0.455 / 0.419     (b) 0.446 / 0.423
    R^2 por componente (a), xi_1..4  oraculo 0.64 0.38 0.20 0.18
                                     lineal  0.64 0.31 0.11 0.12
    R^2 sobre scores ESTIMADOS (c)   oraculo 0.63 0.38 0.16 0.15
                                     lineal  0.62 0.32 0.11 0.11
    varianza entre mecanismos / total, xi_1..4 (a)   0.19 0.05 0.17 0.15
    corr. contemporanea max. (bloque activo) (a)     0.08
    |cos| ejes vs autovectores de Cov(xi) (a)        >= 0.987
    |cos(psi_k, phi_k)|, FPCA estimada (c), k=1..6   0.99 0.99 1.00 0.99 0.99 0.99
    GCV (c): n_basis = 14, orden 4 (K = 14); regla del 95 %: M = 5
    radio espectral de la companera por mecanismo    0.32 0.43 0.43
    sup |xi| bloque activo en 200 000 pasos          2.6
    varianza por bloques de 1000, min / max sobre la media   0.86 / 1.12
    MISE(observada, verdadera) = 0.0625, MISE(suavizada, verdadera) = 0.0121

xi_1 es casi toda persistencia de regimen y un lineal la captura (0.64 contra
0.64): la brecha de C-1 esta en xi_2..xi_4 (0.07-0.09 de R^2 cada una) y, sobre
todo, en la FORMA de la predictiva: cerca de un cambio la ley de xi_1 es
bimodal, algo que ninguna banda gaussiana representa. Es un escenario de
Bloque B antes que de Bloque A.

El oraculo de Bayes
-------------------
Como Z_t depende solo de xi_{t-1..t-L}, observados, y las innovaciones son
independientes,

    E[xi_t | pasado] = sum_k pi_k(xi_{t-1..t-L}) f_k(xi_{t-1..t-L})

es EXACTO, y el proceso es Markov de orden L en xi. Se guarda en
`internos["media_condicional"]` (scores) y `internos["media_condicional_curva"]`
(curva, mu + oraculo @ phi'). El mejor predictor lineal que se reporta es la
regresion por MCO de xi_t sobre los L rezagos de las J componentes, en muestra:
es el techo de un FAR(L) bien especificado en su parte lineal.

`internos["soporte"]` (J, L, J) bool marca con True el par (respuesta j,
rezago l, covariable m) que entra en E[xi_j | pasado]: la union sobre k de los
no nulos de A_{k,l} (o de f_k) y, cuando los mecanismos difieren en la
componente j, las covariables de q_k. `soporte_dinamica` (K, J, L, J) y
`soporte_asignacion` (L, J) guardan las dos partes por separado.

Contrato de reproducibilidad
----------------------------
Por replica se abre UN generador `default_rng(hija)` y se consume, en cada
paso del calentamiento y de las curvas retenidas: el sorteo de Z_t
(`rng.choice`) y luego las J normales de eps_t. Al final, el ruido de
observacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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
from .sim_escenario_B1 import media_seno
from .sim_escenario_C import base_fourier, gram_base

__all__ = [
    "J_SCORES",
    "K_MECANISMOS",
    "BLOQUE_ACTIVO",
    "espectro_nominal",
    "ConfigEscenarioC1",
    "M_TILDE_C1",
    "A_TILDE_C1",
    "C_ASIG_C1",
    "G_ASIG_C1",
    "sigma_tilde",
    "SIGMA_TILDE_C1",
    "MecanismoScores",
    "mecanismo_lineal",
    "radio_espectral_companera",
    "simular_mezcla_scores",
    "generar_mezcla_scores",
    "generar_escenario_C1",
    "resumen_mezcla_scores",
    "resumen_escenario_C1",
]

J_SCORES = 10
K_MECANISMOS = 3
BLOQUE_ACTIVO = 4
AR_PASIVO = 0.3


def espectro_nominal(J: int, lambda_1: float, rho: float) -> np.ndarray:
    """lambda_j = lambda_1 rho^(j-1): varianza nominal de xi_j (s_j^2)."""
    return float(lambda_1) * float(rho) ** np.arange(int(J))


def _softmax(q: np.ndarray) -> np.ndarray:
    q = q - q.max()
    e = np.exp(q)
    return e / e.sum()


# ==========================================================================
# CONFIGURACION
# ==========================================================================

@dataclass
class ConfigEscenarioC1(ConfigObservacion):
    """
    Seccion C del anexo. Los defaults del esquema de observacion son los de las
    corridas vivas (L = 75, T = 1000, sigma_obs = 0.25, R = 1, seed = 41232) y
    burn_in = 300 como en la seccion B.

    J          : scores de la representacion (anexo: 10).
    n_rezagos  : L del anexo (2 en C-1 y C-2, 3 en C-3). No confundir con
                 `L`, que es el tamano de la grilla.
    lambda_1, rho : espectro nominal lambda_j = lambda_1 rho^(j-1).
    """

    L: int = 75
    T: int = 1000
    burn_in: int = 300
    sigma_obs: float = 0.25
    R: int = 1
    seed: int = 41232
    media_fn: Optional[Callable[[np.ndarray], np.ndarray]] = media_seno
    J: int = J_SCORES
    n_rezagos: int = 2
    lambda_1: float = 0.5
    rho: float = 0.55
    prop_train_referencia: float = 0.70

    N_REZAGOS_ANEXO = 2

    def validar(self) -> None:
        super().validar()
        if self.J != J_SCORES:
            raise ValueError(f"J={self.J}: el anexo fija J = {J_SCORES}.")
        if self.n_rezagos != self.N_REZAGOS_ANEXO:
            raise ValueError(f"n_rezagos={self.n_rezagos}: el anexo fija L = "
                             f"{self.N_REZAGOS_ANEXO} para este algoritmo.")
        if self.lambda_1 <= 0 or not 0.0 < self.rho < 1.0:
            raise ValueError("lambda_1 > 0 y rho en (0, 1).")
        if not 0.0 < self.prop_train_referencia < 1.0:
            raise ValueError("prop_train_referencia debe estar en (0, 1).")


# ==========================================================================
# DISENO DE C-1 (coordenadas z; ver docstring del modulo)
# ==========================================================================

def sigma_tilde(activo) -> np.ndarray:
    """sd de eps en z: `activo` en el bloque activo, sqrt(1 - 0.3^2) en el pasivo."""
    pasivo = [np.sqrt(1.0 - AR_PASIVO ** 2)] * (J_SCORES - BLOQUE_ACTIVO)
    return np.array(list(activo) + pasivo, dtype=float)


SIGMA_TILDE_C1 = sigma_tilde((0.45, 0.75, 0.8, 0.8))
"""sd de eps en z de C-1; da Var(z_j) ~ 1 en j = 2..4 (ver docstring)."""

M_TILDE_C1 = np.zeros((K_MECANISMOS, J_SCORES))
M_TILDE_C1[1, 0], M_TILDE_C1[2, 0] = 1.0, -1.0

C_ASIG_C1 = np.array([0.0, -2.1, -2.1])
G_ASIG_C1 = np.zeros((K_MECANISMOS, 2, J_SCORES))
G_ASIG_C1[1, 0, 0], G_ASIG_C1[1, 1, 0], G_ASIG_C1[1, 0, 1] = 1.75, 0.7, 0.5
G_ASIG_C1[2, 0, 0], G_ASIG_C1[2, 1, 0], G_ASIG_C1[2, 0, 1] = -1.75, -0.7, 0.5
"""q_k(x) = C_ASIG[k] + sum_l G_ASIG[k, l] . z_{t-1-l}. Comun a C-1 y C-2."""


def _con_pasivo(A: np.ndarray) -> np.ndarray:
    """Agrega los AR(1) propios del bloque pasivo, iguales en todo k."""
    for j in range(BLOQUE_ACTIVO, A.shape[-1]):
        A[:, 0, j, j] = AR_PASIVO
    return A


def _matrices(entradas: dict, n_rezagos: int) -> np.ndarray:
    """A~ (K, L, J, J) desde {k: [(rezago, respuesta, covariable, valor)]}, base 1."""
    A = np.zeros((K_MECANISMOS, n_rezagos, J_SCORES, J_SCORES))
    for k, lista in entradas.items():
        for l, j, m, v in lista:
            A[k, l - 1, j - 1, m - 1] = v
    return _con_pasivo(A)


_DIAG_C1 = {0: (0.2, 0.3, 0.3, 0.3), 1: (0.2, 0.4, 0.2, 0.2), 2: (0.2, 0.4, 0.2, 0.2)}
_CRUZ_C1 = {  # (2<-1) r1, (3<-2) r1, (1<-1) r2, (4<-3) r2
    0: (0.2, 0.8, -0.1, -0.6),
    1: (0.5, -0.3, 0.1, 0.3),
    2: (-0.5, -0.3, 0.1, 0.3),
}
A_TILDE_C1 = _matrices({
    k: [(1, j + 1, j + 1, _DIAG_C1[k][j]) for j in range(BLOQUE_ACTIVO)]
       + [(1, 2, 1, _CRUZ_C1[k][0]), (1, 3, 2, _CRUZ_C1[k][1]),
          (2, 1, 1, _CRUZ_C1[k][2]), (2, 4, 3, _CRUZ_C1[k][3])]
    for k in range(K_MECANISMOS)
}, 2)
"""A~_{k,l} (K, L, J, J) de C-1: patron comun, valores por mecanismo."""


# ==========================================================================
# MECANISMOS
# ==========================================================================

@dataclass
class MecanismoScores:
    """
    Un algoritmo de la seccion C, en coordenadas z.

    medias(x)      -> (K, J): f~_k(x) para los K mecanismos; x (L, J) con
                      x[l-1] = z_{t-l}.
    asignacion(x)  -> (K,): q~_k(x).
    sigma          : (J,) sd de la innovacion en z.
    soporte_dinamica   : (K, J, L, J) bool, pares que entran en f~_k.
    soporte_asignacion : (L, J) bool, pares que entran en q~.
    difiere        : (J,) bool, componentes donde f~_k no es la misma en todo k.
    matrices, medias_const : A~ y mu~ si el mecanismo es lineal; None si no.
    jacobiano      : (K, L, J, J) linealizacion de f~_k en x = 0, para el radio
                     espectral cuando f~_k no es lineal (C-3).
    """

    medias: Callable[[np.ndarray], np.ndarray]
    asignacion: Callable[[np.ndarray], np.ndarray]
    sigma: np.ndarray
    n_rezagos: int
    soporte_dinamica: np.ndarray
    soporte_asignacion: np.ndarray
    difiere: np.ndarray
    matrices: Optional[np.ndarray] = None
    medias_const: Optional[np.ndarray] = None
    jacobiano: Optional[np.ndarray] = None

    def soporte(self) -> np.ndarray:
        """(J, L, J): union sobre k de f~_k mas q~ donde los mecanismos difieren."""
        S = self.soporte_dinamica.any(axis=0)
        S[self.difiere] |= self.soporte_asignacion[None, :, :]
        return S


def mecanismo_lineal(M_tilde: np.ndarray, A_tilde: np.ndarray,
                     c: np.ndarray, G: np.ndarray,
                     sigma: np.ndarray) -> MecanismoScores:
    """C-1 / C-2: f~_k(x) = mu~_k + sum_l A~_{k,l} x_l, q~_k(x) = c_k + sum_l G_{k,l} . x_l."""
    L = A_tilde.shape[1]

    def medias(x):
        return M_tilde + np.einsum("klij,lj->ki", A_tilde, x)

    def asignacion(x):
        return c + np.einsum("klj,lj->k", G, x)

    iguales = np.array([
        all(np.array_equal(A_tilde[k, :, j], A_tilde[0, :, j])
            and M_tilde[k, j] == M_tilde[0, j] for k in range(A_tilde.shape[0]))
        for j in range(A_tilde.shape[2])])
    return MecanismoScores(
        medias=medias, asignacion=asignacion, sigma=np.asarray(sigma, float),
        n_rezagos=L, soporte_dinamica=(A_tilde != 0).transpose(0, 2, 1, 3),
        soporte_asignacion=(G != 0).any(axis=0), difiere=~iguales,
        matrices=A_tilde, medias_const=M_tilde)


def radio_espectral_companera(A: np.ndarray) -> float:
    """Radio espectral de la companera de (A_1, ..., A_L), A (L, J, J)."""
    L, J, _ = A.shape
    C = np.zeros((L * J, L * J))
    C[:J] = np.hstack(list(A))
    if L > 1:
        C[J:, :-J] = np.eye(J * (L - 1))
    return float(np.max(np.abs(np.linalg.eigvals(C))))


# ==========================================================================
# MOTOR COMUN A C-1, C-2 Y C-3
# ==========================================================================

def simular_mezcla_scores(mec: MecanismoScores, T: int, burn_in: int,
                          rng: np.random.Generator) -> dict:
    """
    Itera, en coordenadas z,

        x = (z_{t-1}, ..., z_{t-L}),  pi_t = softmax(q~(x)),  Z_t ~ Cat(pi_t),
        z_t = f~_{Z_t}(x) + sigma * e_t,

    desde z = 0, y devuelve los T periodos retenidos: z (T, J), Z (T,),
    pi (T, K), el oraculo sum_k pi_k f~_k(x) (T, J) y la varianza entre
    mecanismos sum_k pi_k (f~_k - oraculo)^2 (T, J).
    """
    L, K = mec.n_rezagos, K_MECANISMOS
    J = mec.sigma.size
    n = int(burn_in) + int(T)
    z = np.zeros((n + L, J))
    Z = np.empty(n, dtype=int)
    P = np.empty((n, K))
    ora = np.empty((n, J))
    ventre = np.empty((n, J))
    for i in range(n):
        t = i + L
        x = z[t - L:t][::-1]
        p = _softmax(np.asarray(mec.asignacion(x), dtype=float))
        k = int(rng.choice(K, p=p))
        F = np.asarray(mec.medias(x), dtype=float)
        z[t] = F[k] + mec.sigma * rng.standard_normal(J)
        o = p @ F
        Z[i], P[i], ora[i] = k, p, o
        ventre[i] = p @ (F - o) ** 2
    c = int(burn_in)
    return {"z": z[L + c:], "Z": Z[c:], "pi": P[c:], "oraculo": ora[c:],
            "var_entre": ventre[c:]}


def generar_mezcla_scores(cfg: ConfigEscenarioC1, mec: MecanismoScores,
                          diagnosticar: Callable[[SalidaSimulacion], dict]) -> SalidaSimulacion:
    """Genera las R replicas de un algoritmo C y arma `SalidaSimulacion`."""
    cfg.validar()
    tau = grilla_regular(int(cfg.L))
    mu = evaluar_media(cfg.media_fn, tau)
    Phi = base_fourier(tau, int(cfg.J))                    # (L, J)
    lam = espectro_nominal(cfg.J, cfg.lambda_1, cfg.rho)
    s = np.sqrt(lam)
    # Coeficientes de mu en la base (exactos: mu cae en su espacio). Con ellos la
    # representacion en la base es theta = mu_theta + xi (sin estimar nada).
    mu_theta = np.linalg.solve(gram_base(Phi, tau), Phi.T @ (pesos_trapezoidales(tau) * mu))

    hijas, registro = semillas_replicas(cfg.seed, cfg.R)
    R_, T, G, J = int(cfg.R), int(cfg.T), int(cfg.L), int(cfg.J)
    obs = np.empty((R_, T, G))
    curvas = np.empty((R_, T, G))
    mc_curva = np.empty((R_, T, G))
    scores = np.empty((R_, T, J))
    mc = np.empty((R_, T, J))
    ventre = np.empty((R_, T, J))
    Z_all = np.empty((R_, T), dtype=int)
    P_all = np.empty((R_, T, K_MECANISMOS))

    for r, hija in enumerate(hijas):
        rng = np.random.default_rng(hija)
        sim = simular_mezcla_scores(mec, T, int(cfg.burn_in), rng)
        scores[r] = sim["z"] * s
        mc[r] = sim["oraculo"] * s
        ventre[r] = sim["var_entre"] * s ** 2
        Z_all[r], P_all[r] = sim["Z"], sim["pi"]
        curvas[r] = mu[None, :] + scores[r] @ Phi.T
        mc_curva[r] = mu[None, :] + mc[r] @ Phi.T
        obs[r] = aplicar_ruido_observacion(curvas[r], cfg.sigma_obs, rng)

    internos = {
        "Phi": Phi,
        "lambda": lam,
        "escala_scores": s,
        "sigma_innovacion": mec.sigma * s,
        "mecanismo": Z_all,
        "pi": P_all,
        "scores": scores,
        "mu_theta": mu_theta,
        "theta": mu_theta[None, None, :] + scores,
        "media_condicional": mc,
        "media_condicional_curva": mc_curva,
        "varianza_entre_mecanismos": ventre,
        "soporte": mec.soporte(),
        "soporte_dinamica": mec.soporte_dinamica,
        "soporte_asignacion": mec.soporte_asignacion,
        "n_rezagos": np.array(mec.n_rezagos),
    }
    if mec.matrices is not None:
        internos["medias_mecanismos"] = mec.medias_const * s[None, :]
        internos["matrices_dependencia"] = (
            s[None, None, :, None] * mec.matrices / s[None, None, None, :])
    lin = mec.matrices if mec.matrices is not None else mec.jacobiano
    if lin is not None:
        internos["radio_espectral"] = np.array(
            [radio_espectral_companera(lin[k]) for k in range(K_MECANISMOS)])

    salida = SalidaSimulacion(
        observaciones=obs, curvas=curvas, grilla=tau, media=mu,
        semillas=registro, config=cfg, internos=internos,
    )
    salida.diagnostico = diagnosticar(salida)
    return salida


# ==========================================================================
# CONTROL DE CALIDAD COMUN
# ==========================================================================

def _rachas(Z: np.ndarray) -> np.ndarray:
    """Largo medio de las rachas de cada mecanismo."""
    cambios = np.flatnonzero(np.diff(Z) != 0)
    fin = np.r_[cambios, len(Z) - 1]
    largos = np.diff(np.r_[-1, fin])
    return np.array([largos[Z[fin] == k].mean() if (Z[fin] == k).any() else np.nan
                     for k in range(K_MECANISMOS)])


def _r2_lineal(xi: np.ndarray, L: int) -> np.ndarray:
    """R^2 por componente de MCO de xi_t sobre los L rezagos de las J componentes."""
    n = len(xi)
    y = xi[L:]
    X = np.column_stack([np.ones(n - L)] + [xi[L - l:n - l] for l in range(1, L + 1)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1.0 - (y - X @ b).var(0) / y.var(0), ((y - X @ b) ** 2).sum(0)


def resumen_mezcla_scores(salida: SalidaSimulacion) -> dict:
    """
    Control de calidad comun a C-1, C-2 y C-3, promediado sobre replicas:
    ocupacion y racha media de cada mecanismo, espectro realizado y alineacion
    de sus autovectores con los ejes (la FPCA poblacional de los scores
    verdaderos, sin ruido ni base), R^2 del oraculo de Bayes y del mejor
    predictor lineal en los L rezagos (por componente y agregado en L^2),
    fraccion de varianza entre mecanismos, acotamiento y estabilidad por
    bloques. Los R^2 son en muestra sobre los T periodos retenidos.
    """
    cfg = salida.config
    I = salida.internos
    L = int(I["n_rezagos"])
    A = BLOQUE_ACTIVO
    Zs, P, XI, MC, VE = I["mecanismo"], I["pi"], I["scores"], I["media_condicional"], \
        I["varianza_entre_mecanismos"]
    R_ = XI.shape[0]

    occ, rach, ntr, r2o, r2l, r2o_l2, r2l_l2 = [], [], [], [], [], [], []
    ev_all, cos_all, corr_max, fr_entre, var_bloques = [], [], [], [], []
    for r in range(R_):
        occ.append(np.bincount(Zs[r], minlength=K_MECANISMOS) / Zs.shape[1])
        rach.append(_rachas(Zs[r]))
        ntr.append(int((np.diff(Zs[r]) != 0).sum()))
        xi = XI[r]
        y = xi[L:] - xi[L:].mean(0)
        sst = (y ** 2).sum(0)
        sse_o = ((xi[L:] - MC[r, L:]) ** 2).sum(0)
        r2l_r, sse_l = _r2_lineal(xi, L)
        r2o.append(1.0 - sse_o / sst)
        r2l.append(r2l_r)
        r2o_l2.append(1.0 - sse_o.sum() / sst.sum())
        r2l_l2.append(1.0 - sse_l.sum() / sst.sum())
        C = np.cov(xi.T)
        ev, U = np.linalg.eigh(C)
        o = np.argsort(ev)[::-1]
        ev_all.append(ev[o])
        cos_all.append(np.abs(np.diag(U[:, o])))
        Cr = np.corrcoef(xi[:, :A].T)
        corr_max.append(float(np.abs(Cr - np.eye(A)).max()))
        fr_entre.append(VE[r].mean(0) / xi.var(0))
        var_bloques.append(np.array([b.var(0)[:A].sum() for b in np.array_split(xi, 4)]))

    ev = np.mean(ev_all, axis=0)
    vb = np.mean(var_bloques, axis=0)
    rad = I.get("radio_espectral")
    G = gram_base(I["Phi"], salida.grilla)
    return {
        **diagnostico_comun(salida),
        "n_rezagos": L,
        "ocupacion_mecanismos": np.mean(occ, axis=0).tolist(),
        "racha_media_mecanismos": np.nanmean(rach, axis=0).tolist(),
        "n_transiciones": float(np.mean(ntr)),
        "pi_media": P.reshape(-1, K_MECANISMOS).mean(0).tolist(),
        "pi_desviacion_del_centro": float(np.abs(P - 1.0 / K_MECANISMOS).mean()),
        "var_scores": XI.reshape(-1, XI.shape[-1]).var(0).tolist(),
        "lambda_nominal": I["lambda"].tolist(),
        "var_acum_scores": (np.cumsum(ev) / ev.sum()).tolist(),
        "cos_ejes_scores": np.mean(cos_all, axis=0).tolist(),
        "corr_contemporanea_max_activo": float(np.mean(corr_max)),
        "r2_oraculo_por_componente": np.mean(r2o, axis=0)[:A + 1].tolist(),
        "r2_lineal_por_componente": np.mean(r2l, axis=0)[:A + 1].tolist(),
        "r2_oraculo": float(np.mean(r2o_l2)),
        "r2_lineal": float(np.mean(r2l_l2)),
        "brecha_no_lineal": float(np.mean(r2o_l2) - np.mean(r2l_l2)),
        "fraccion_varianza_entre_mecanismos": np.mean(fr_entre, axis=0)[:A].tolist(),
        "max_abs_scores_activo": float(np.abs(XI[..., :A]).max()),
        "var_bloques_activo": vb.tolist(),
        "var_bloques_razon_max_min": float(vb.max() / vb.min()),
        "radio_espectral_mecanismos": (rad.tolist() if rad is not None else None),
        "error_ortonormalidad": float(np.abs(G - np.eye(G.shape[0])).max()),
        "T0_referencia": int(np.floor(cfg.prop_train_referencia * cfg.T)),
    }


# ==========================================================================
# GENERADOR C-1
# ==========================================================================

def mecanismo_C1() -> MecanismoScores:
    return mecanismo_lineal(M_TILDE_C1, A_TILDE_C1, C_ASIG_C1, G_ASIG_C1, SIGMA_TILDE_C1)


def generar_escenario_C1(cfg: ConfigEscenarioC1) -> SalidaSimulacion:
    """R replicas del Algoritmo C-1."""
    return generar_mezcla_scores(cfg, mecanismo_C1(), resumen_escenario_C1)


def resumen_escenario_C1(salida: SalidaSimulacion) -> dict:
    if not isinstance(salida.config, ConfigEscenarioC1):
        raise TypeError("resumen_escenario_C1 requiere ConfigEscenarioC1; se "
                        f"recibio {type(salida.config).__name__}.")
    return resumen_mezcla_scores(salida)
