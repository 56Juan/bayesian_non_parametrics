"""
far_operador.py
===============
Estimador FAR(1) de Bosq (2000) en la formulacion efectiva del paquete `far`
de R (Damon & Guillas, version 0.6-7), como OPERADOR sobre las curvas.

    X_{t+1}(tau) = rho(X_t)(tau) + eps_{t+1}(tau),   X_t en L^2[0, 1].

Por que existe este modulo
--------------------------
El repositorio ya tiene algo llamado "FAR1": `ajustar_far1`, dentro del
notebook `_05_comparacion`. Ese estimador calcula `Psi = C1 C0^-1` con
`C0 = A'A/n` y `C1 = B'A/n` sobre los SCORES FPCA estandarizados. Esos momentos
estan SIN centrar, de modo que equivalen a minimos cuadrados sin intercepto
sobre M coordenadas; y como los scores ya vienen centrados sobre el bloque de
entrenamiento y MCO es equivariante bajo cambios afines de escala, el resultado
coincide con el VAR sobre scores hasta `corr = 1.00000000` con M=1 y
`0.99999486` con M=2 (medido en la corrida 20). Es decir: no es una segunda
referencia lineal, es la misma con otro nombre.

La diferencia metodologica que aqui se recupera no esta en la formula
`C1 C0^-1` sino en DONDE vive y COMO se regulariza:

    ajustar_far1   opera sobre M scores FPCA ya elegidos por la regla del 95 %
                   de varianza acumulada, y no regulariza (`RIDGE_FAR = 0`).

    FAR1 (aqui)    opera sobre la curva en la grilla, estima su propia base
                   propia a partir de C0 y trunca en `kn` dimensiones, con `kn`
                   elegido por error de PREDICCION y no por varianza explicada.

`kn` y `M` no son la misma cantidad y no deben confundirse
----------------------------------------------------------
`M` es el numero de componentes FPCA que el pipeline retiene para representar
la curva, y sale de la regla del 95 % de varianza acumulada
(`fpca.seleccionar_M(0.95)`). `kn` es el orden de truncamiento espectral con
que se REGULARIZA la inversion de C0 al estimar el operador. Son criterios
distintos porque responden a preguntas distintas: la varianza acumulada mide
cuanto de la curva se conserva, mientras que el problema que aqui se resuelve
esta mal planteado ---C0^-1 no es un operador acotado--- y lo que hay que
controlar es la amplificacion del error de estimacion en las direcciones de
autovalor pequeno. Una componente puede aportar poca varianza y mucha
prediccion, o mucha varianza y ninguna. Por eso `kn` se elige por error de
prediccion a un paso (`seleccionar_kn`) y nunca por porcentaje de varianza.

Especificacion: el comportamiento efectivo de far::far()
--------------------------------------------------------
Lo que sigue NO se reconstruyo de la teoria general de Bosq sino que se leyo
del fuente de `far_0.6-7` (`R/far.R`, `R/invgen.R`, `R/fdata.R`) y se verifico
numericamente contra la salida de R (ver `comparar_con_r`). Con una sola
variable (`y="x"`, sin `x`, `joined=FALSE`) el estimador es:

    Xc     = X - media puntual sobre las n curvas          (centrado unico)
    C0     = Xc Xc' / n                                    (normaliza por n)
    V      = primeros kn autovectores de C0                (L x kn)
    S      = V' Xc                                         (scores, kn x n)
    Delta  = suma_t s_t s_{t-1}'                           (suma CRUDA)
    G      = suma_u s_u s_u'                               (suma CRUDA)
    rho    = Delta pinv(G) * nbobs / nbobs2                (kn x kn)

Puntos que solo se ven en el fuente y que aqui se replican:

1.  La normalizacion es irrelevante. `Delta` y `G` son sumas sin dividir y el
    factor `nbobs/nbobs2` corrige exactamente el desbalance entre las n curvas
    que entran en C0 y los n-1 pares que entran en C1, de modo que
    `rho = C1_hat C0_hat^-1` sin ambiguedad entre n y n-1.
2.  El centrado usa UNA sola media ---la curva media puntual sobre las n
    columnas--- restada tanto al regresor como a la respuesta, y se vuelve a
    sumar al predecir (`predict.far`). No hay dos medias distintas.
3.  La regularizacion es truncamiento espectral a `kn` autovectores de C0
    aplicado POR LOS DOS LADOS (los scores son kn-dimensionales), y encima una
    pseudo-inversa de Moore-Penrose por SVD con `tol = sqrt(eps)` relativo al
    mayor valor singular. Son dos redes de seguridad superpuestas, no una.
4.  `object$rho` NO es el nucleo en la grilla: es kn x kn, en la base propia.
    El operador sobre la grilla es `V rho V'`, como se ve en `predict.far`
    (`pred = V %*% (rho %*% (V' x))`). Es una matriz de ACCION, directamente
    comparable con `internos["operador"]` de los generadores del repositorio,
    que tambien lo es (nucleo por pesos de cuadratura).
5.  `far` NO usa pesos de cuadratura en ninguna parte: trabaja con `crossprod`
    sobre los vectores discretizados, es decir con la medida de conteo. Ver la
    seccion siguiente, porque el asunto es mas sutil de lo que parece.
6.  `na.rm` descarta la columna ---la curva entera--- ante cualquier NA, y un
    par requiere ambos extremos presentes.
7.  `kn` como vector significa "un valor por variable" en el caso
    multivariado. No es un barrido. Aqui solo se implementa el caso univariado.

Rarezas de `far` que NO se replican, y por que
----------------------------------------------
`far` guarda en `$values` los autovalores DIVIDIDOS por el numero de puntos de
la grilla, y `print.far` calcula su "Explained Variance" sobre los autovalores
AL CUADRADO. Ninguna de las dos cantidades es la varianza explicada usual, de
modo que los porcentajes que imprime R no son comparables con los del FPCA de
este repositorio. Aqui `valores_propios_` son los autovalores de C0 sin
reescalar. La estimacion de `rho` no depende de esto en absoluto.

Ademas, `as.fdata` etiqueta por defecto la grilla como `(0:(p-1))/p`, es decir
sin el extremo derecho, mientras que el esquema de observacion del estudio usa
`tau_1 = 0, ..., tau_L = 1`. Son solo etiquetas y el estimador de R no las usa,
pero conviene saberlo al cruzar figuras.

Los pesos de cuadratura se cancelan, salvo en el truncamiento
-------------------------------------------------------------
Esta es la unica divergencia numerica genuina entre esta implementacion y R, y
merece el detalle porque la intuicion enganya. Escribiendo las matrices de
accion con `W = diag(w)` los pesos de cuadratura:

    rho = (suma y_t y_{t-1}' W) (suma y y' W)^-1 = suma y_t y_{t-1}' (suma y y')^-1

`W` desaparece del producto. La matriz de accion estimada es LIBRE DE PESOS, y
en eso R ---que usa medida de conteo--- y una implementacion en L^2 coinciden
exactamente. La ausencia de cuadratura en `far` no es un error.

Donde `W` no se cancela es en el TRUNCAMIENTO: los autovectores de `Xc Xc'`
(ortonormales en el producto interno euclideo) no son los de `Xc Xc' W`
(ortonormales en L^2), y `V` si entra en el estimador a traves de la proyeccion
a `kn` dimensiones. Sobre una grilla regular la diferencia se reduce al medio
peso de los dos extremos, de modo que es pequenya y esta CONCENTRADA EN LOS
BORDES DEL DOMINIO. Por eso `comparar_con_r` reporta el error maximo puntual
ademas del relativo en norma: un promedio lo esconde justamente donde ocurre.

Consecuencia de diseno: `pesos` es un parametro. Por defecto sale de
`utils.quadrature.pesos_trapezoidales`, que es la unica definicion de la regla
de integracion del proyecto y no se redefine aqui. Con `pesos="conteo"` se
usan pesos unitarios y la clase reproduce a `far::far()` hasta precision de
maquina; eso es un test (`tests/test_far_operador.py`), no una aspiracion.

Sensibilidad a kn: el problema esta mal planteado
--------------------------------------------------
`C0^-1` no es acotado, de modo que el estimador es discontinuo en el limite y
`kn` no es un hiperparametro cosmetico: gobierna el sesgo contra la varianza.
Con `kn` grande las direcciones de autovalor pequenyo entran divididas por un
numero cercano a cero y amplifican el ruido de estimacion; con `kn` chico el
operador se estima bien pero sobre un subespacio que puede no contener la
direccion que gobierna la dinamica. `diagnostico_kn()` reporta el autovalor
minimo retenido y el numero de condicion efectivo `lambda_1 / lambda_kn`, que
es la cifra que anticipa la inestabilidad antes de que se vea en el error.

Ausencia de fuga
----------------
`fit` recibe UNICAMENTE el bloque de entrenamiento. La media, la base propia y
`rho` se estiman ahi y `predict` los aplica sin recalcular nada, de modo que la
ausencia de fuga es una propiedad de la clase y no una disciplina del notebook
---el mismo patron de `FPCA` y del estandarizador---. `seleccionar_kn` divide
el bloque de entrenamiento otra vez y nunca toca el de prueba.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..utils.quadrature import pesos_trapezoidales

__all__ = [
    "FAR1",
    "ResultadoCV",
    "seleccionar_kn",
    "simular_far1",
    "comparar_con_r",
    "comparar_con_ajustar_far1",
    "norma_hs",
]


# ==========================================================================
# UTILIDADES DE ALGEBRA
# ==========================================================================

def _resolver_pesos(
    pesos: Union[str, np.ndarray, None],
    grilla: Optional[np.ndarray],
    L: int,
) -> np.ndarray:
    """
    Devuelve el vector de pesos de cuadratura (L,).

    `pesos` admite:
      None / "trapecio"  pesos trapezoidales sobre `grilla` (por defecto).
      "conteo"           pesos unitarios: la medida de conteo que usa
                         far::far(). Es lo que hay que pasar para reproducir R
                         hasta precision de maquina.
      array (L,)         pesos arbitrarios, para grillas no regulares o reglas
                         de orden superior.

    No se redefine aqui la regla trapezoidal: se importa de
    `utils.quadrature`, que es su unica definicion en el proyecto.
    """
    if isinstance(pesos, str):
        if pesos == "conteo":
            return np.ones(L, dtype=float)
        if pesos != "trapecio":
            raise ValueError(
                f"pesos='{pesos}' no reconocido; use 'trapecio', 'conteo' o "
                "un arreglo (L,)."
            )
        pesos = None
    if pesos is None:
        if grilla is None:
            raise ValueError(
                "Con pesos trapezoidales hay que pasar `grilla`; para "
                "reproducir far::far() use pesos='conteo'."
            )
        w = pesos_trapezoidales(np.asarray(grilla, dtype=float))
    else:
        w = np.asarray(pesos, dtype=float).ravel()
    if w.size != L:
        raise ValueError(f"Los pesos tienen {w.size} entradas y la grilla {L}.")
    if np.any(w <= 0):
        raise ValueError("Todos los pesos de cuadratura deben ser positivos.")
    return w


def _pinv_far(G: np.ndarray) -> np.ndarray:
    """
    Pseudo-inversa con el mismo criterio de corte que `far::invgen`.

    R usa `La.svd` y descarta los valores singulares por debajo de
    `sqrt(.Machine$double.eps) * d[1]`. `np.linalg.pinv` usa por defecto un
    corte distinto (`max(M, N) * eps`), de modo que fijarlo explicitamente no
    es cosmetico: con autovalores pequenyos ---exactamente el regimen en que
    `kn` importa--- los dos criterios retienen conjuntos distintos.
    """
    return np.linalg.pinv(G, rcond=np.sqrt(np.finfo(float).eps))


def norma_hs(A: np.ndarray, pesos: np.ndarray) -> float:
    """
    Norma de Hilbert-Schmidt de un operador integral dado por su MATRIZ DE
    ACCION `A` (L, L), es decir la que satisface `(rho f)(tau_l) ~= (A f)_l`.

    El nucleo es `k = A W^-1`, y su norma de Hilbert-Schmidt discretizada es

        ||rho||_HS^2 = suma_{l,l'} w_l w_l' k(tau_l, tau_l')^2.

    Se escribe asi ---y no como norma de Frobenius de `A`--- porque es la
    convencion con que `sim_comun.matriz_operador_ar` calibra `hs_norm`: el
    operador guardado en `internos["operador"]` de los generadores ya lleva los
    pesos incorporados. Comparar Frobenius contra esta cantidad da un error de
    orden `h` que se confunde con un fallo del estimador.
    """
    A = np.asarray(A, dtype=float)
    w = np.asarray(pesos, dtype=float)
    k = A / w[None, :]
    return float(np.sqrt(np.sum((w[:, None] * w[None, :]) * k ** 2)))


# ==========================================================================
# EL ESTIMADOR
# ==========================================================================

class FAR1:
    """
    Estimador FAR(1) como operador sobre las curvas, replicando far::far().

    Parametros
    ----------
    kn : int
        Orden del truncamiento espectral de C0. Es el UNICO hiperparametro y
        gobierna el compromiso sesgo/varianza de un problema mal planteado:
        subirlo amplia el subespacio donde el operador puede vivir y a la vez
        divide por autovalores cada vez mas pequenyos. No es el `M` del FPCA
        (ver el docstring del modulo). Con `kn=None` hay que llamar a
        `seleccionar_kn` antes de `fit`, o pasar `kn` explicito.
    grilla : (L,), opcional
        Puntos del dominio. Necesaria para los pesos trapezoidales por defecto.
    pesos : "trapecio" | "conteo" | (L,), opcional
        Regla de cuadratura. Ver `_resolver_pesos`. El efecto esperado de
        cambiarla es NULO sobre `rho_` salvo por el subespacio retenido, y ese
        efecto se concentra en los bordes del dominio.
    center : bool
        Resta la curva media estimada en `fit`. `far::far()` usa `TRUE` por
        defecto y es lo razonable salvo que las curvas ya vengan centradas: sin
        centrar, `rho` tiene que explicar tambien el nivel medio y lo hace mal,
        porque el nivel no es una direccion de variacion del proceso.

    Atributos ajustados
    -------------------
    media_ : (L,)                curva media del bloque de entrenamiento
    base_ : (L, kn)              autovectores retenidos de C0
    valores_propios_ : (L,)      autovalores de C0, sin reescalar
    rho_subespacio_ : (kn, kn)   el `$rho` de R, en la base propia
    rho_ : (L, L)                matriz de ACCION sobre la grilla, `V rho V'`
    """

    def __init__(
        self,
        kn: int = 2,
        grilla: Optional[np.ndarray] = None,
        pesos: Union[str, np.ndarray, None] = None,
        center: bool = True,
    ) -> None:
        if kn is not None and (not isinstance(kn, (int, np.integer)) or kn < 1):
            raise ValueError(f"kn={kn}: debe ser un entero >= 1.")
        self.kn = int(kn)
        self.grilla = None if grilla is None else np.asarray(grilla, dtype=float)
        self.pesos = pesos
        self.center = bool(center)

    # ----------------------------------------------------------------------

    def fit(self, X: np.ndarray, na_rm: bool = True) -> "FAR1":
        """
        Estima el operador con `X` de forma (T, L): filas = curvas, columnas =
        puntos de la grilla.

        Notese la transposicion respecto de R, donde `fdata` almacena las
        curvas con filas = grilla y columnas = observaciones. Se adopta la
        convencion del repositorio ---(T, L), la de `X_curves.npy`--- y la
        transposicion ocurre una sola vez, aqui. Es el mismo tipo de error
        silencioso que motivo `utils/trazas.py` al cruzar Python y MATLAB, y
        por eso `comparar_con_r` lo verifica con una matriz asimetrica.

        SOLO se le pasa el bloque de entrenamiento. La clase no conoce `T0`
        justamente para que no pueda mirar mas alla de lo que recibe.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D (T, L); recibido {X.shape}.")
        T, L = X.shape
        if T < 3:
            raise ValueError(f"Hacen falta al menos 3 curvas; recibidas {T}.")
        if self.kn > min(L, T):
            raise ValueError(
                f"kn={self.kn} excede min(L, T) = {min(L, T)}: no hay tantas "
                "direcciones estimables."
            )

        w = _resolver_pesos(self.pesos, self.grilla, L)
        self.pesos_ = w

        # Curvas utilizables. `far` descarta la curva entera ante cualquier NA.
        if na_rm:
            ok = np.all(np.isfinite(X), axis=1)
        else:
            ok = np.ones(T, dtype=bool)
        if ok.sum() < 3:
            raise ValueError("Quedan menos de 3 curvas completas tras na_rm.")
        # Un par (t-1, t) requiere ambos extremos presentes.
        par = np.zeros(T, dtype=bool)
        par[1:] = ok[1:] & ok[:-1]
        nbobs, nbobs2 = int(ok.sum()), int(par.sum())
        if nbobs2 < 2:
            raise ValueError("No hay suficientes pares consecutivos completos.")
        self.n_curvas_, self.n_pares_ = nbobs, nbobs2

        # Centrado: una sola media, la puntual sobre las curvas utilizables.
        self.media_ = X[ok].mean(axis=0) if self.center else np.zeros(L)
        Xc = X - self.media_[None, :]
        Xc = np.where(np.isfinite(Xc), Xc, 0.0)

        # C0 como operador: la matriz de accion es (1/n) sum y y' W. Se
        # diagonaliza el problema generalizado en la metrica de los pesos
        # simetrizando con W^{1/2}, de modo que la base resulte ortonormal en
        # la metrica elegida. Con pesos unitarios esto se reduce EXACTAMENTE
        # al `eigen(Xc Xc'/n)` de R.
        raiz_w = np.sqrt(w)
        Y = Xc[ok] * raiz_w[None, :]                    # (nbobs, L)
        C0_sim = (Y.T @ Y) / nbobs                      # simetrica definida >= 0
        valores, vectores = np.linalg.eigh(C0_sim)
        orden = np.argsort(valores)[::-1]
        valores, vectores = valores[orden], vectores[:, orden]
        self.valores_propios_ = valores
        # Vuelta a la escala original: v = W^{-1/2} u, ortonormal en <.,.>_w.
        V = vectores[:, : self.kn] / raiz_w[:, None]     # (L, kn)
        self.base_ = V

        # Scores. El producto interno lleva los pesos, de modo que
        # S = V' W Xc'. Con pesos unitarios coincide con el `t(v) %*% data`
        # de R.
        S = (Xc * w[None, :]) @ V                        # (T, kn)
        self.scores_ = S

        # rho en el subespacio. Sumas crudas y el factor nbobs/nbobs2, tal
        # como en el fuente: asi la normalizacion se cancela y no hay que
        # decidir entre n y n-1.
        Sp, Sf = S[:-1][par[1:]], S[1:][par[1:]]         # predictor, respuesta
        Delta = Sf.T @ Sp                                # sum_t s_t s_{t-1}'
        G = S[ok].T @ S[ok]                              # sum_u s_u s_u'
        self.rho_subespacio_ = Delta @ _pinv_far(G) * (nbobs / nbobs2)

        # Matriz de ACCION sobre la grilla. El paso a la grilla lleva los
        # pesos por el lado del argumento: (rho f)(tau) = V rho (V' W f).
        self.rho_ = V @ self.rho_subespacio_ @ (V * w[:, None]).T
        return self

    # ----------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prediccion a un paso: dada `X` (n, L) con las curvas en t, devuelve
        (n, L) con las curvas predichas en t+1.

        No es recursivo. Cada fila se predice con su propio rezago observado,
        que es el conjunto de informacion con que trabajan los demas
        competidores del `_05` y la condicion para que la comparacion sea
        entre modelos y no entre horizontes.
        """
        self._verificar_ajustado()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.rho_.shape[0]:
            raise ValueError(
                f"X tiene {X.shape[1]} puntos y el operador se ajusto con "
                f"{self.rho_.shape[0]}."
            )
        Xc = X - self.media_[None, :]
        return Xc @ self.rho_.T + self.media_[None, :]

    def predict_serie(self, X: np.ndarray) -> np.ndarray:
        """
        Prediccion a un paso a lo largo de una serie `X` (T, L): devuelve
        (T-1, L), la prediccion de la curva `t` hecha con la curva `t-1`.

        Es la forma que consume la ventana movil y la que hay que usar para
        construir la fila del `_05`: alinea con `X[1:]`.
        """
        return self.predict(np.asarray(X, dtype=float)[:-1])

    def simulate(
        self,
        n: int,
        cov_innovacion: Optional[np.ndarray] = None,
        sigma: float = 1.0,
        burn_in: int = 200,
        x0: Optional[np.ndarray] = None,
        rng: Union[int, np.random.Generator, None] = None,
    ) -> np.ndarray:
        """
        Simula `n` curvas del FAR(1) AJUSTADO, devolviendo (n, L).

        Con `cov_innovacion=None` la innovacion es ruido blanco puntual de
        desviacion `sigma`, que NO es una innovacion funcional suave y solo
        sirve para pruebas de estabilidad del operador. Para generar datos con
        verdad de terreno usar `simular_far1`, o mejor el generador del
        repositorio (ver su docstring).
        """
        self._verificar_ajustado()
        rng = np.random.default_rng(rng)
        L = self.rho_.shape[0]
        if cov_innovacion is None:
            chol = sigma * np.eye(L)
        else:
            from ..utils.linalg import safe_chol
            chol = safe_chol(np.asarray(cov_innovacion, dtype=float))
        Y = np.zeros(L) if x0 is None else np.asarray(x0, float) - self.media_
        salida = np.empty((n, L))
        for t in range(burn_in + n):
            Y = self.rho_ @ Y + chol @ rng.standard_normal(L)
            if t >= burn_in:
                salida[t - burn_in] = self.media_ + Y
        return salida

    # ----------------------------------------------------------------------

    def norma_hs(self) -> float:
        """||rho_hat||_HS con la convencion de `sim_comun.matriz_operador_ar`."""
        self._verificar_ajustado()
        return norma_hs(self.rho_, self.pesos_)

    def diagnostico_kn(self) -> dict:
        """
        Cifras que anticipan la inestabilidad debida al mal planteamiento.

        `lambda_min_retenido` es el autovalor por el que se esta dividiendo en
        la direccion mas debil del subespacio, y `condicion` su cociente con el
        mayor. Un `condicion` grande no invalida el ajuste, pero significa que
        `rho_` es sensible a `kn` y que la comparacion entre puntos del barrido
        mezcla el efecto de la dinamica con el de la regularizacion.
        `varianza_retenida` se reporta SOLO como descripcion: no es el criterio
        de seleccion, y ponerlo al lado del 95 % del FPCA seria confundir dos
        cosas distintas.
        """
        self._verificar_ajustado()
        val = self.valores_propios_
        val_pos = val[val > 0]
        return {
            "kn": self.kn,
            "lambda_1": float(val[0]),
            "lambda_min_retenido": float(val[self.kn - 1]),
            "condicion": float(val[0] / val[self.kn - 1])
            if val[self.kn - 1] > 0 else np.inf,
            "varianza_retenida": float(val[: self.kn].sum() / val_pos.sum()),
            "norma_hs": self.norma_hs(),
            "n_curvas": self.n_curvas_,
            "n_pares": self.n_pares_,
        }

    def _verificar_ajustado(self) -> None:
        if not hasattr(self, "rho_"):
            raise RuntimeError("El estimador no esta ajustado; llame a fit().")


# ==========================================================================
# SELECCION DE kn
# ==========================================================================

@dataclass
class ResultadoCV:
    """
    Salida de `seleccionar_kn`.

    `tabla` tiene una fila por `kn` con los seis criterios de `far.cv`, y
    `kn_L2` es el elegido por el criterio L2, que es el que se usa por defecto
    por ser el unico coherente con el MISE que reporta el resto del capitulo.
    """
    tabla: np.ndarray            # (n_kn, 7): kn, L1, L2, Linf, L1max, L2max, Linfmax
    columnas: tuple
    kn_L1: int
    kn_L2: int
    kn_Linf: int
    n_validacion: int

    @property
    def kn(self) -> int:
        return self.kn_L2


def seleccionar_kn(
    X_train: np.ndarray,
    kn_max: int,
    n_validacion: Optional[int] = None,
    grilla: Optional[np.ndarray] = None,
    pesos: Union[str, np.ndarray, None] = None,
    center: bool = True,
) -> ResultadoCV:
    """
    Elige `kn` por error de prediccion a un paso, replicando `far::far.cv`.

    IMPORTANTE, y es la razon de que esta funcion exista: `far.cv` NO es
    validacion cruzada por particiones. Es un HOLD-OUT del bloque final: ajusta
    con las primeras `n - ncv` curvas y evalua predicciones a un paso sobre las
    ultimas `ncv`, con `ncv = round(n/5)` por defecto. Que sea un corte
    temporal y no una particion aleatoria es lo correcto en una serie ---una
    particion aleatoria filtraria futuro en el ajuste--- pero significa que el
    criterio se estima sobre un solo bloque y por tanto es ruidoso. Con
    `T0 = 280` y `ncv = 56` es una cifra de varianza apreciable: conviene mirar
    la tabla completa y no solo el argmin, porque el minimo suele ser plano.

    El `X_train` que recibe es SOLO el bloque de entrenamiento del estudio: el
    hold-out se abre DENTRO de el y el bloque de prueba nunca se toca.

    Los seis criterios son los de R: L1, L2 y Linf del error, promediados sobre
    las curvas del hold-out, y los mismos tres sobre el maximo de cada curva
    (`*max`), que miden el error en el pico y no en promedio.
    """
    X_train = np.asarray(X_train, dtype=float)
    n = X_train.shape[0]
    if n_validacion is None:
        n_validacion = int(round(n / 5))
    if n_validacion < 2 or n_validacion >= n - 2:
        raise ValueError(
            f"n_validacion={n_validacion} no deja bloques utilizables con n={n}."
        )
    n1 = n - n_validacion
    aprende, valida = X_train[:n1], X_train[n1:]
    if kn_max > min(X_train.shape[1], n1):
        raise ValueError(
            f"kn_max={kn_max} excede min(L, n_ajuste) = {min(X_train.shape[1], n1)}."
        )

    filas = []
    for k in range(1, kn_max + 1):
        mod = FAR1(kn=k, grilla=grilla, pesos=pesos, center=center).fit(aprende)
        err = valida[1:] - mod.predict(valida[:-1])      # (ncv-1, L)
        pico = valida[1:].max(axis=1) - mod.predict(valida[:-1]).max(axis=1)
        filas.append((
            k,
            np.mean(np.abs(err)),                        # L1
            np.mean(np.sqrt(np.mean(err ** 2, axis=1))), # L2
            np.mean(np.max(np.abs(err), axis=1)),        # Linf
            np.mean(np.abs(pico)),                       # L1max
            np.sqrt(np.mean(pico ** 2)),                 # L2max
            np.max(np.abs(pico)),                        # Linfmax
        ))
    tabla = np.array(filas, dtype=float)
    return ResultadoCV(
        tabla=tabla,
        columnas=("kn", "L1", "L2", "Linf", "L1max", "L2max", "Linfmax"),
        kn_L1=int(tabla[np.argmin(tabla[:, 1]), 0]),
        kn_L2=int(tabla[np.argmin(tabla[:, 2]), 0]),
        kn_Linf=int(tabla[np.argmin(tabla[:, 3]), 0]),
        n_validacion=int(n_validacion),
    )


# ==========================================================================
# SIMULADOR CON VERDAD DE TERRENO
# ==========================================================================

def simular_far1(
    L: int = 75,
    T: int = 400,
    c_kernel: Optional[float] = None,
    hs_objetivo: float = 0.70,
    sigma: float = 1.0,
    burn_in: int = 200,
    seed: Union[int, np.random.Generator, None] = 41232,
) -> dict:
    """
    Simulador minimo de FAR(1) con nucleo `rho(t, s) = c exp(-(t^2 + s^2)/2)`.

    ADVERTENCIA DE ALCANCE, y es deliberada: para las pruebas de consistencia
    del estudio hay que usar `pipelines.generar_escenario_1`, NO esta funcion.
    El Algoritmo 1 del anexo ya es un FAR(1) lineal gaussiano con operador de
    nucleo gaussiano, `||Psi||_HS = 0.70` calibrada, innovacion funcional suave
    con Cholesky estable y el operador verdadero guardado en `internos`; es
    decir, ya provee verdad de terreno y duplicar su maquinaria de cuadratura y
    factorizacion es exactamente la clase de duplicacion que este repositorio
    ya pago una vez. Esta funcion existe solo porque el nucleo separable
    `exp(-(t^2+s^2)/2)` es de RANGO UNO, y eso la hace util para una prueba que
    el generador del repositorio no da: verificar que el estimador recupera un
    operador cuyo rango es conocido y menor que `kn`, donde el truncamiento no
    introduce sesgo y la convergencia debe ser limpia.

    La innovacion es ruido blanco puntual, no funcional. Por eso NO sirve para
    evaluar el pipeline: sus curvas no son suaves y el FPCA sobre ellas no
    tiene el espectro decreciente que el estudio supone.

    Devuelve un dict con `curvas` (T, L), `grilla`, `operador` (matriz de
    accion, comparable con `rho_`) y `hs`.
    """
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, 1.0, L)
    w = pesos_trapezoidales(tau)
    nucleo = np.exp(-(tau[:, None] ** 2 + tau[None, :] ** 2) / 2.0)
    hs_bruto = float(np.sqrt(np.sum((w[:, None] * w[None, :]) * nucleo ** 2)))
    c = hs_objetivo / hs_bruto if c_kernel is None else float(c_kernel)
    A = c * nucleo * w[None, :]                     # matriz de accion
    hs = norma_hs(A, w)
    if hs >= 1.0:
        raise ValueError(
            f"||rho||_HS = {hs:.4f} >= 1: el proceso no seria estacionario."
        )
    Y = np.zeros(L)
    curvas = np.empty((T, L))
    for t in range(burn_in + T):
        Y = A @ Y + sigma * rng.standard_normal(L)
        if t >= burn_in:
            curvas[t - burn_in] = Y
    return {"curvas": curvas, "grilla": tau, "operador": A, "hs": hs,
            "pesos": w, "c": c}


# ==========================================================================
# COMPARACIONES
# ==========================================================================

def comparar_con_r(
    X: np.ndarray,
    rho_r: np.ndarray,
    kn: int,
    grilla: Optional[np.ndarray] = None,
    pesos: Union[str, np.ndarray, None] = "conteo",
    center: bool = True,
) -> dict:
    """
    Contrasta esta implementacion contra el rho_hat de far::far().

    `rho_r` debe ser el operador SOBRE LA GRILLA, es decir `V %*% rho %*% t(V)`
    calculado en R a partir de `aj$v[[1]]` y `aj$rho` ---no `aj$rho` a secas,
    que es kn x kn en la base propia---.

    ADVERTENCIA DE TRANSPOSICION. R es column-major y 1-based, y `fdata`
    almacena las curvas con filas = grilla y columnas = observaciones, es decir
    TRASPUESTO respecto del (T, L) de este repositorio. Antes de concluir que
    hay una diferencia metodologica hay que descartar que sea de layout, y por
    eso esta funcion reporta `error_max_traspuesto` junto al directo: si el
    traspuesto es el pequenyo, el problema es de orientacion y no del metodo.
    La verificacion solo es concluyente con una matriz ASIMETRICA; con una
    simetrica los dos errores coinciden y no dicen nada, y por eso se reporta
    tambien `asimetria`, que es lo que hace valido el diagnostico.

    Por defecto `pesos="conteo"`, que es la medida que usa R. Con pesos
    trapezoidales la comparacion sigue siendo informativa pero ya no debe dar
    cero: la diferencia esta concentrada en los bordes del dominio (ver el
    docstring del modulo) y es la razon de que se reporte el error maximo
    puntual ademas del relativo en norma.
    """
    X = np.asarray(X, dtype=float)
    rho_r = np.asarray(rho_r, dtype=float)
    mod = FAR1(kn=kn, grilla=grilla, pesos=pesos, center=center).fit(X)
    rho_py = mod.rho_
    if rho_py.shape != rho_r.shape:
        raise ValueError(
            f"Formas incompatibles: python {rho_py.shape}, R {rho_r.shape}."
        )

    dif = rho_py - rho_r
    escala = float(np.abs(rho_r).max())
    asimetria = float(np.abs(rho_r - rho_r.T).max())

    # Correlacion entre las predicciones a un paso de ambos operadores, que es
    # la cantidad que de verdad importa: dos operadores pueden diferir en
    # direcciones que los datos nunca visitan.
    Xc = X - mod.media_[None, :]
    p_py = Xc[:-1] @ rho_py.T
    p_r = Xc[:-1] @ rho_r.T
    corr = float(np.corrcoef(p_py.ravel(), p_r.ravel())[0, 1])

    return {
        "error_rel_L2": float(np.linalg.norm(dif) / np.linalg.norm(rho_r)),
        "error_max_puntual": float(np.abs(dif).max()),
        "error_max_relativo": float(np.abs(dif).max() / escala) if escala else np.nan,
        "error_max_traspuesto": float(np.abs(rho_py - rho_r.T).max()),
        "asimetria_rho_r": asimetria,
        "layout_verificable": bool(asimetria > 10 * np.abs(dif).max()),
        "corr_predicciones": corr,
        "kn": kn,
        "pesos": pesos if isinstance(pesos, str) else "arreglo",
    }


def comparar_con_ajustar_far1(
    pred_far: np.ndarray,
    pred_referencia: np.ndarray,
    etiqueta: str = "VAR",
) -> dict:
    """
    Cuantifica cuanto se SEPARA este estimador de otro, con la metrica que ya
    usa el repositorio para el par (VAR, FAR1): maximo absoluto de la
    diferencia relativo a la desviacion tipica del ajuste, y correlacion.

    Ambos argumentos son (n, M) en la misma escala. La cifra que decide es
    `max_relativo_sd`: el `_05` considera que dos estimadores no se separan por
    debajo de `TOL_VAR_FAR = 2 %`. Si este estimador tampoco supera ese umbral,
    la conclusion es que el capitulo sigue teniendo UNA sola referencia lineal
    y hay que decirlo con el numero delante, no meterlo igual.
    """
    a = np.asarray(pred_far, dtype=float).ravel()
    b = np.asarray(pred_referencia, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Formas distintas: {a.shape} y {b.shape}.")
    sd = float(np.std(b))
    dmax = float(np.abs(a - b).max())
    return {
        "referencia": etiqueta,
        "max_abs": dmax,
        "sd_ajuste": sd,
        "max_relativo_sd": dmax / sd if sd > 0 else np.nan,
        "corr": float(np.corrcoef(a, b)[0, 1]),
    }
