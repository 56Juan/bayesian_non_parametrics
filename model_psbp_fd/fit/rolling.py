r"""
rolling.py — Evolucion de las metricas sobre una ventana movil
==============================================================

Implementa el segundo componente del eje 1 del diseno de simulacion
(`docs/03 Modelo.tex §03_06`): la evolucion del error a lo largo del tiempo,
sobre una ventana que recorre el bloque de entrenamiento y el de prueba.

Que es y que NO es
------------------
La ventana movil NO reentrena el modelo. El muestreador se ejecuta una sola
vez con el bloque {1, ..., T0} y nunca vuelve a ver los datos. Lo que se
desliza es la ventana de EVALUACION: para cada origen t se dispone de una
prediccion a horizonte h=1 construida con los rezagos REALES en t --nunca con
predicciones encadenadas-- y la ventana agrega los errores de w origenes
consecutivos en una sola cifra,

    RMSE_w(t) = sqrt( (1/w) sum_{u=t-w+1}^{t} e_u ),   t = w, ..., T,

que se grafica contra t con una marca en T0. La serie de errores por origen es
demasiado ruidosa para leerse directamente; la ventana la convierte en una
curva interpretable sin introducir ningun supuesto adicional.

Por que es informativa
----------------------
Es la unica vista en que entrenamiento y prueba comparten eje, y por eso separa
tres cosas que las metricas agregadas confunden:

    el SALTO en T0        cuantifica la degradacion fuera de muestra. Plano
                          significa que el modelo generaliza; un salto grande
                          es sobreajuste.
    la DERIVA dentro de   delata no estacionariedad. En el Escenario 1, que es
    cada bloque           el control FAR(1) estacionario, la curva debe ser
                          plana a ambos lados y sin salto: ese resultado es la
                          referencia contra la cual se leen los demas.
    los EPISODIOS         en el Escenario 2 la ventana debe mostrar los brotes
    locales               de volatilidad y en el 3 los cambios de regimen. Un
                          w demasiado grande los difumina, y por eso w es un
                          parametro y no una constante.

Dentro del bloque de entrenamiento la prediccion es in-sample: el modelo ya vio
esos datos. No es fuga --no se usa para elegir nada-- y es precisamente el
contraste con el bloque de prueba lo que hace informativa la figura.

Sobre que se calcula
--------------------
`ventana_movil_scores`   opera sobre los scores, una serie por componente FPCA.
`ventana_movil_funcional` opera sobre las curvas, una sola serie que agrega las
                          M componentes en la metrica L^2. Es la cantidad que
                          le importa a la tesis.

En simulacion la curva de referencia es la VERDADERA del generador, no la
observada: el ruido de medicion sigma_eps no forma parte de lo que el modelo
debe predecir, y compararse contra los datos lo cuenta como error del modelo.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics_distribucional import crps_muestral
from .metrics_puntual import _2d
from ..utils.quadrature import pesos_trapezoidales

__all__ = [
    "indices_ventanas",
    "ventana_movil",
    "ventana_movil_scores",
    "ventana_movil_funcional",
]


# ==========================================================================
# GEOMETRIA DE LAS VENTANAS
# ==========================================================================

def indices_ventanas(n: int, w: int, paso: int = 1,
                     solapadas: bool = True) -> List[np.ndarray]:
    """
    Indices (base-0) de cada ventana de ancho `w` sobre una serie de largo `n`.

    solapadas=True  : ventanas deslizantes, una cada `paso` origenes. Es el
                      modo por defecto porque la resolucion de la figura la da
                      el deslizamiento y no el tamano de la muestra: con 120
                      origenes de prueba y w=20 hay 101 posiciones.
    solapadas=False : bloques disjuntos consecutivos. Util cuando cada cifra
                      debe leerse como una estimacion independiente y no como
                      una curva suavizada.
    """
    n, w, paso = int(n), int(w), int(paso)
    if w < 1:
        raise ValueError(f"w={w} debe ser al menos 1.")
    if w > n:
        raise ValueError(f"w={w} excede el largo de la serie n={n}.")
    if paso < 1:
        raise ValueError(f"paso={paso} debe ser al menos 1.")

    salto = w if not solapadas else paso
    return [np.arange(ini, ini + w)
            for ini in range(0, n - w + 1, salto)]


def _bloque(t_centro: int, T0: int) -> str:
    """Etiqueta del bloque al que pertenece el origen `t_centro` (base-0)."""
    return "train" if t_centro < T0 else "test"


# ==========================================================================
# NUCLEO GENERICO
# ==========================================================================

def ventana_movil(metricas: Dict[str, Callable[[np.ndarray], float]],
                  n: int, T0: int, w: int, paso: int = 1,
                  solapadas: bool = True,
                  t_offset: int = 0) -> pd.DataFrame:
    """
    Aplica un diccionario de metricas a cada ventana y devuelve una tabla larga.

    metricas : {nombre: f(idx) -> float}, donde `idx` son los indices base-0 de
        los origenes de la ventana dentro de la serie completa de largo `n`.
        Se recibe la funcion y no los errores ya calculados porque hay metricas
        --cobertura, CRPS, energy score-- que no son promedios de un error por
        origen y no pueden reconstruirse a partir de uno.
    n  : numero de origenes de la serie evaluada.
    T0 : corte train/test en el MISMO indexado que `n`.
    t_offset : desplazamiento para reportar `t` en el tiempo del experimento
        cuando la serie evaluada no empieza en t=0 (p. ej. tras descartar los
        primeros N_LAGS origenes).

    Columnas: t_ini, t_fin, t_centro (tiempo del experimento, base-1), bloque,
    n_ventana, y una columna por metrica.
    """
    filas = []
    for idx in indices_ventanas(n, w, paso, solapadas):
        t_ini, t_fin = int(idx[0]), int(idx[-1])
        t_centro = (t_ini + t_fin) // 2
        fila = {
            "t_ini":     t_ini + t_offset + 1,       # base-1, tiempo experimento
            "t_fin":     t_fin + t_offset + 1,
            "t_centro":  t_centro + t_offset + 1,
            "bloque":    _bloque(t_centro, T0),
            "n_ventana": int(idx.size),
        }
        # Una ventana puede cruzar T0. Se etiqueta por su centro y se marca,
        # porque su cifra mezcla dentro y fuera de muestra y no debe leerse
        # como ninguno de los dos.
        fila["cruza_T0"] = bool(t_ini < T0 <= t_fin)
        for nombre, f in metricas.items():
            fila[nombre] = float(f(idx))
        filas.append(fila)

    if not filas:
        raise ValueError(f"Ninguna ventana cabe: n={n}, w={w}.")
    return pd.DataFrame(filas)


# ==========================================================================
# VENTANA MOVIL SOBRE LOS SCORES
# ==========================================================================

def ventana_movil_scores(y_obs: np.ndarray, y_pred: np.ndarray, T0: int,
                         w: int, paso: int = 1, solapadas: bool = True,
                         t_offset: int = 0,
                         muestras: Optional[np.ndarray] = None,
                         li: Optional[np.ndarray] = None,
                         ls: Optional[np.ndarray] = None,
                         etiquetas: Optional[Sequence[str]] = None
                         ) -> pd.DataFrame:
    """
    Evolucion del error por componente FPCA.

    y_obs, y_pred : (n, M) observado y predicho a h=1, en la MISMA escala.
    T0    : corte train/test en el indexado de `y_obs`.
    muestras : (S, n, M) opcional. Si se entrega se agrega `crps` a la tabla.
    li, ls   : (n, M) opcional. Si se entregan se agrega `cobertura` y `ancho`.

    Retorna una tabla larga con una fila por (ventana, componente).
    """
    Y, P = _2d(y_obs), _2d(y_pred)
    if Y.shape != P.shape:
        raise ValueError(f"y_obs {Y.shape} y y_pred {P.shape} no coinciden.")
    n, M = Y.shape
    nombres = list(etiquetas) if etiquetas is not None else [
        f"fpc_{m + 1}" for m in range(M)]

    partes = []
    for m in range(M):
        err2 = (Y[:, m] - P[:, m]) ** 2
        metricas: Dict[str, Callable[[np.ndarray], float]] = {
            "rmse": lambda idx, e=err2: float(np.sqrt(e[idx].mean())),
            "mae":  lambda idx, y=Y[:, m], p=P[:, m]: float(
                np.abs(y[idx] - p[idx]).mean()),
            # R2 centrado DENTRO de la ventana: mide si el modelo bate a la
            # media local, que es el rival honesto a esa escala temporal. El R2
            # global usa la media global y premia por capturar nivel, no
            # dinamica.
            "r2_local": lambda idx, y=Y[:, m], e=err2: float(
                1.0 - e[idx].sum()
                / max(float(((y[idx] - y[idx].mean()) ** 2).sum()), 1e-12)),
            "sd_obs":  lambda idx, y=Y[:, m]: float(y[idx].std(ddof=1)),
            "sd_pred": lambda idx, p=P[:, m]: float(p[idx].std(ddof=1)),
        }
        if muestras is not None:
            Z = np.asarray(muestras, dtype=float)
            if Z.ndim != 3 or Z.shape[1:] != (n, M):
                raise ValueError(
                    f"muestras debe ser (S, {n}, {M}); recibido {Z.shape}.")
            metricas["crps"] = lambda idx, z=Z[:, :, m], y=Y[:, m]: float(
                crps_muestral(y[idx], z[:, idx]).mean())
        if li is not None and ls is not None:
            L, U = _2d(li), _2d(ls)
            metricas["cobertura"] = lambda idx, y=Y[:, m], a=L[:, m], b=U[:, m]: float(
                np.mean((y[idx] >= a[idx]) & (y[idx] <= b[idx])))
            metricas["ancho"] = lambda idx, a=L[:, m], b=U[:, m]: float(
                np.mean(b[idx] - a[idx]))

        tabla = ventana_movil(metricas, n, T0, w, paso, solapadas, t_offset)
        tabla.insert(0, "componente", nombres[m])
        partes.append(tabla)

    salida = pd.concat(partes, ignore_index=True)
    salida.attrs["w"] = int(w)
    salida.attrs["solapadas"] = bool(solapadas)
    return salida


# ==========================================================================
# VENTANA MOVIL SOBRE LAS CURVAS
# ==========================================================================

def ventana_movil_funcional(X_obs: np.ndarray, X_pred: np.ndarray,
                            tau: np.ndarray, T0: int, w: int, paso: int = 1,
                            solapadas: bool = True, t_offset: int = 0,
                            li: Optional[np.ndarray] = None,
                            ls: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Evolucion del error funcional: una sola serie que agrega las M componentes.

    X_obs  : (n, G) curvas de referencia. En simulacion son las VERDADERAS del
        generador, sin ruido de medicion. Compararse contra las observadas
        atribuye sigma_eps al modelo y desplaza toda la curva hacia arriba por
        una razon que no depende de nada que el modelo pueda hacer mejor.
    X_pred : (n, G) curvas predichas a h=1.
    tau    : (G,) grilla. La integral usa la cuadratura trapezoidal comun del
        proyecto, no una suma simple: con grilla regular la diferencia es el
        peso 1/2 en los extremos, y duplicarla en otra parte del codigo es
        justamente lo que `utils.quadrature` existe para evitar.
    li, ls : (n, G) opcional, banda puntual. Agrega `cobertura_puntual`, que es
        la fraccion de pares (t, tau_g) cubiertos: NO es cobertura simultanea
        de la curva y debe declararse asi al reportar.

    Metricas por ventana:
        mise   : (1/w) sum_t integral (X_t - Xhat_t)^2 dtau
        rmse_f : sqrt(mise)
        mise_rel : mise dividido por la varianza funcional de X_obs en la
                   ventana. Adimensional, de modo que las ventanas con distinta
                   amplitud de senal son comparables entre si y entre
                   escenarios.
    """
    O, P = _2d(X_obs), _2d(X_pred)
    if O.shape != P.shape:
        raise ValueError(f"X_obs {O.shape} y X_pred {P.shape} no coinciden.")
    tau = np.asarray(tau, dtype=float).ravel()
    if tau.size != O.shape[1]:
        raise ValueError(f"tau tiene {tau.size} puntos y las curvas {O.shape[1]}.")

    n = O.shape[0]
    pesos = pesos_trapezoidales(tau)                       # (G,)
    err_t = ((O - P) ** 2) @ pesos                         # (n,) integral por t

    def _mise(idx):
        return float(err_t[idx].mean())

    def _mise_rel(idx):
        centro = O[idx] - O[idx].mean(axis=0, keepdims=True)
        var_f = float((centro ** 2 @ pesos).mean())
        return float(err_t[idx].mean() / max(var_f, 1e-12))

    metricas: Dict[str, Callable[[np.ndarray], float]] = {
        "mise":     _mise,
        "rmse_f":   lambda idx: float(np.sqrt(err_t[idx].mean())),
        "mise_rel": _mise_rel,
    }
    if li is not None and ls is not None:
        L, U = _2d(li), _2d(ls)
        dentro = (O >= L) & (O <= U)                       # (n, G)
        metricas["cobertura_puntual"] = lambda idx: float(dentro[idx].mean())
        metricas["ancho_medio"] = lambda idx: float((U[idx] - L[idx]).mean())

    salida = ventana_movil(metricas, n, T0, w, paso, solapadas, t_offset)
    salida.attrs["w"] = int(w)
    salida.attrs["solapadas"] = bool(solapadas)
    return salida
