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

Horizonte
---------
Todo lo que hay aqui es a HORIZONTE h = 1, con los rezagos REALES en cada
origen y nunca con predicciones encadenadas. La `h` de la especificacion de
metricas --MAE(h), RMSE(h)-- es por tanto constante e igual a 1 en todo el
recorrido, y no existe ninguna ponderacion sobre horizontes: la unica
ponderacion viva es la del DOMINIO FUNCIONAL, que viaja como `pesos_tau` y esta
documentada en `metrics_puntual.pesos_normalizados`. Los dos objetos se nombran
distinto a proposito para que no puedan confundirse el dia que se agreguen
horizontes: extender a h > 1 exigiria decidir antes entre iterado --propagar la
predictiva muestra a muestra, que es lo unico coherente con el Bloque B-- y
directo --un ajuste por horizonte--, y ninguna de las dos cosas esta
implementada.

Bloque A y Bloque B sobre la ventana
------------------------------------
`ventana_movil_funcional` emite, ademas del MISE de siempre:

    Bloque A (error puntual, normas del mismo error e_t(tau)):
        mae_f      ||e||_1 promediado sobre los origenes de la ventana
        rmse_f     RAIZ DEL MSE AGREGADO, sqrt(mean_t int e_t^2 dw)
        l2_medio   PROMEDIO de ||e_t||_2, que NO es lo mismo (Jensen)
        linf_max, linf_medio, q95_abs, razon_linf_l1

    Bloque B (intervalos; solo si se entregan li y ls):
        winkler    PRIMARIA del bloque, regla de puntuacion propia
        picp, mpiw DIAGNOSTICAS: la descomposicion del Winkler en cobertura y
                   ancho. Nunca rankean modelos por si solas.

ORDEN DE AGREGACION, que la especificacion de metricas exige cerrar: `rmse_f`
es y sigue siendo `sqrt(mise)`, es decir la raiz del MSE agregado sobre
(origenes, tau), y NO el promedio de los RMSE por origen. Las dos cifras
difieren --por Jensen la segunda es menor-- y ambas se emiten: `l2_medio` es la
segunda, y su razon con `rmse_f` mide cuan desigual es el error entre los
origenes de la ventana. Se dejan las dos justamente para que la eleccion sea
visible en la tabla en vez de quedar enterrada en una formula.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics_distribucional import crps_muestral, winkler
from .metrics_puntual import _2d, normas_error_por_origen, pesos_normalizados
from ..utils.quadrature import pesos_trapezoidales
from ..utils.progreso import Progreso

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


def _columna(a) -> np.ndarray:
    """
    Lleva a (n, M) una entrada que puede venir aplanada por el punto M = 1.

    NO es `_2d`, y la diferencia importa: `np.atleast_2d` interpreta un vector
    (n,) como UNA FILA, es decir un solo origen con n componentes, que es
    exactamente al reves de lo que ocurre en el punto M = 1 del barrido. Ahi un
    vector plano es una SERIE de n origenes con una componente, y el repositorio
    lo produce por tres caminos distintos ya documentados: `np.loadtxt` colapsa
    a 1D con una sola columna, MATLAB elimina el eje singleton final al guardar
    los `.mat`, y sklearn devuelve (n,) cuando el objetivo tiene una columna.
    Con `_2d` la ventana movil no fallaba con una cifra rara --fallaba con
    "w excede el largo de la serie n=1"--, pero el diagnostico no apuntaba a la
    causa. Aqui la forma se arregla y el caso queda cubierto por
    `tests/test_metricas_bloques_AB.py`.
    """
    A = np.asarray(a, dtype=float)
    return A[:, None] if A.ndim == 1 else np.atleast_2d(A)


# ==========================================================================
# NUCLEO GENERICO
# ==========================================================================

def ventana_movil(metricas: Dict[str, Callable[[np.ndarray], float]],
                  n: int, T0: int, w: int, paso: int = 1,
                  solapadas: bool = True,
                  t_offset: int = 0,
                  verbose: bool = False,
                  etiqueta: str = "ventana_movil") -> pd.DataFrame:
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
    verbose : informa el avance sobre las ventanas. Con `crps` o `energy` entre
        las metricas cada ventana recorre las S extracciones de la predictiva,
        y con ventanas solapadas hay del orden de `n` de ellas: es el bucle mas
        caro del pipeline de error. No afecta al resultado.
    etiqueta : nombre que encabeza las lineas de progreso. Lo fijan las
        funciones que envuelven a esta para que se distinga la pasada por
        scores de la pasada funcional, y para que en el barrido por componente
        se vea cual se esta procesando.

    Columnas: t_ini, t_fin, t_centro (tiempo del experimento, base-1), bloque,
    n_ventana, y una columna por metrica.
    """
    ventanas = indices_ventanas(n, w, paso, solapadas)
    prog = Progreso(etiqueta, total=len(ventanas), verbose=verbose)

    filas = []
    for idx in ventanas:
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
        # Se muestra la primera metrica del diccionario como testigo: basta
        # para ver que las cifras son del orden esperado y no NaN.
        primera = next(iter(metricas), None)
        prog.paso(f"t={fila['t_centro']} [{fila['bloque']}]"
                  + (f" {primera}={fila[primera]:.4g}" if primera else ""))

    if not filas:
        prog.fin("ninguna ventana")
        raise ValueError(f"Ninguna ventana cabe: n={n}, w={w}.")
    tabla = pd.DataFrame(filas)
    prog.fin(f"{int(tabla['cruza_T0'].sum())} ventanas cruzan T0")
    return tabla


# ==========================================================================
# VENTANA MOVIL SOBRE LOS SCORES
# ==========================================================================

def ventana_movil_scores(y_obs: np.ndarray, y_pred: np.ndarray, T0: int,
                         w: int, paso: int = 1, solapadas: bool = True,
                         t_offset: int = 0,
                         muestras: Optional[np.ndarray] = None,
                         li: Optional[np.ndarray] = None,
                         ls: Optional[np.ndarray] = None,
                         etiquetas: Optional[Sequence[str]] = None,
                         nivel: float = 0.95,
                         verbose: bool = False
                         ) -> pd.DataFrame:
    """
    Evolucion del error por componente FPCA.

    y_obs, y_pred : (n, M) observado y predicho a h=1, en la MISMA escala. Un
        vector (n,) se admite y se lee como n origenes de UNA componente, que
        es la forma que toman las cosas en el punto M = 1 del barrido.
    T0    : corte train/test en el indexado de `y_obs`.
    muestras : (S, n, M) opcional. Si se entrega se agrega `crps` a la tabla.
    li, ls   : (n, M) opcional. Si se entregan se agregan `cobertura` y `ancho`
        y, con los mismos numeros, sus nombres del Bloque B `picp` y `mpiw`,
        mas `winkler`, que es la metrica PRIMARIA de ese bloque.
    nivel : nivel nominal de (li, ls). Solo lo usa el Winkler, cuya
        penalizacion por fallo es (2/alpha) veces la distancia al intervalo: si
        no coincide con el nivel con que se construyo la banda, la cifra deja
        de ser interpretable.
    verbose : informa el avance ventana a ventana, con la componente en curso
        en la etiqueta. El costo es M veces el de `ventana_movil`, y con
        `muestras` cada ventana calcula el CRPS sobre las S extracciones.

    Retorna una tabla larga con una fila por (ventana, componente).
    """
    Y, P = _columna(y_obs), _columna(y_pred)
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
            # (S, n) es la forma que devuelve el muestreador en el punto M = 1.
            if Z.ndim == 2 and M == 1 and Z.shape[1] == n:
                Z = Z[:, :, None]
            if Z.ndim != 3 or Z.shape[1:] != (n, M):
                raise ValueError(
                    f"muestras debe ser (S, {n}, {M}); recibido {Z.shape}.")
            # El CRPS se calcula UNA vez para toda la serie y las ventanas
            # solo promedian: es una cifra POR ORIGEN, de modo que restringir
            # las columnas antes o despues da exactamente el mismo numero. Con
            # ventanas solapadas la diferencia de costo es de dos ordenes
            # -habia un ordenamiento de S x w valores por ventana, y hay uno de
            # S x n en total-, y con T grande era el cuello de botella del
            # notebook de evaluacion.
            crps_t = crps_muestral(Y[:, m], Z[:, :, m])          # (n,)
            metricas["crps"] = lambda idx, c=crps_t: float(c[idx].mean())
        if li is not None and ls is not None:
            L, U = _columna(li), _columna(ls)
            # `cobertura` y `ancho` se conservan con sus nombres de siempre
            # --hay figuras y CSV que los buscan asi-- y al lado van sus
            # sinonimos del Bloque B, `picp` y `mpiw`, mas la metrica PRIMARIA
            # del bloque, que es el Winkler. Duplicar dos columnas es el precio
            # de no romper lo que ya consume esta tabla.
            w_t = winkler(Y[:, m], L[:, m], U[:, m], nivel=nivel)     # (n,)
            metricas["cobertura"] = lambda idx, y=Y[:, m], a=L[:, m], b=U[:, m]: float(
                np.mean((y[idx] >= a[idx]) & (y[idx] <= b[idx])))
            metricas["ancho"] = lambda idx, a=L[:, m], b=U[:, m]: float(
                np.mean(b[idx] - a[idx]))
            metricas["winkler"] = lambda idx, s=w_t: float(s[idx].mean())
            metricas["picp"] = metricas["cobertura"]
            metricas["mpiw"] = metricas["ancho"]

        tabla = ventana_movil(
            metricas, n, T0, w, paso, solapadas, t_offset,
            verbose=verbose,
            etiqueta=f"ventana_movil_scores[{nombres[m]}, w={w}]")
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
                            ls: Optional[np.ndarray] = None,
                            pesos_tau: Optional[np.ndarray] = None,
                            q_extremo: float = 0.95,
                            nivel: float = 0.95,
                            bloque_A: bool = True,
                            verbose: bool = False) -> pd.DataFrame:
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
        de la curva y debe declararse asi al reportar. Con `bloque_A` agrega
        ademas `winkler`, `picp` y `mpiw`.
    pesos_tau : ponderacion opcional del dominio funcional, la MISMA para todos
        los modelos que se comparan (ver `metrics_puntual.pesos_normalizados`).
        Solo afecta a las metricas del Bloque A; `mise` conserva la cuadratura
        sin normalizar de siempre para no cambiar cifras ya reportadas.
    q_extremo : orden del cuantil de |e| sobre el dominio que sustituye al
        supremo como cifra citable del peor caso.
    nivel : nivel nominal de la banda; lo usa el Winkler.
    bloque_A : emite las normas L^p del error y, si hay banda, el Bloque B.
        En False la tabla es exactamente la de antes de esta extension.
    verbose : informa el avance ventana a ventana. No afecta al resultado.

    Metricas por ventana:
        mise   : (1/w) sum_t integral (X_t - Xhat_t)^2 dtau
        rmse_f : sqrt(mise)  <-- RAIZ DEL MSE AGREGADO, no promedio de RMSE
        mise_rel : mise dividido por la varianza funcional de X_obs en la
                   ventana. Adimensional, de modo que las ventanas con distinta
                   amplitud de senal son comparables entre si y entre
                   escenarios.

    Con `bloque_A=True` se agregan, en este orden de lectura:
        mae_f      norma L^1 del error, promediada sobre los origenes de la
                   ventana. Estima la mediana condicional y es robusta.
        l2_medio   promedio de ||e_t||_2 por origen. NO coincide con `rmse_f`:
                   por Jensen l2_medio <= rmse_f, con igualdad solo si el error
                   es constante entre origenes. Se emiten las dos a proposito.
        razon_agregacion = l2_medio / rmse_f, en (0, 1]. Proxima a uno el error
                   es homogeneo dentro de la ventana; baja, hay origenes que
                   dominan el agregado.
        linf_max, linf_medio, q95_abs, razon_linf_l1  (peor caso y
                   concentracion del error; `q95_abs` es la cifra citable y
                   `linf_max` la fragil, que depende de una sola evaluacion).
        winkler, picp, mpiw  solo con banda. `winkler` es la PRIMARIA del
                   Bloque B; las otras dos son su descomposicion y no rankean.

    La cadena ||e||_1 <= ||e||_2 <= ||e||_inf se verifica con assert dentro de
    `normas_error_por_origen`, antes de que ninguna de estas cifras llegue a la
    tabla.
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
        # sqrt del MSE agregado sobre (origenes, tau). Declarado en el
        # docstring porque NO es el promedio de los RMSE por origen.
        "rmse_f":   lambda idx: float(np.sqrt(err_t[idx].mean())),
        "mise_rel": _mise_rel,
    }

    if bloque_A:
        # Las tres normas se calculan UNA vez para toda la serie --son cifras
        # por origen-- y las ventanas solo las indexan. Es lo que permite
        # engancharse por el diccionario de callables sin tocar el motor: cada
        # metrica del Bloque A es una media sobre `idx` de un vector ya
        # calculado. La cadena L1 <= L2 <= Linf se verifica ahi dentro.
        nm = normas_error_por_origen(O, P, tau, pesos_tau=pesos_tau,
                                     q=q_extremo, verificar=True)
        _qn = f"q{int(round(q_extremo * 100))}_abs"
        metricas.update({
            "mae_f":       lambda idx, v=nm["l1"]:   float(v[idx].mean()),
            "l2_medio":    lambda idx, v=nm["l2"]:   float(v[idx].mean()),
            "linf_max":    lambda idx, v=nm["linf"]: float(v[idx].max()),
            "linf_medio":  lambda idx, v=nm["linf"]: float(v[idx].mean()),
            _qn:           lambda idx, v=nm["q_abs"]: float(v[idx].mean()),
            "razon_linf_l1": lambda idx, v=nm["razon_linf_l1"]: float(v[idx].mean()),
            "razon_agregacion": lambda idx, a=nm["l2"]: float(
                a[idx].mean() / max(float(np.sqrt((a[idx] ** 2).mean())), 1e-300)),
        })

    if li is not None and ls is not None:
        L, U = _2d(li), _2d(ls)
        dentro = (O >= L) & (O <= U)                       # (n, G)
        metricas["cobertura_puntual"] = lambda idx: float(dentro[idx].mean())
        metricas["ancho_medio"] = lambda idx: float((U[idx] - L[idx]).mean())
        if bloque_A:
            # Winkler integrado sobre el dominio con los pesos NORMALIZADOS,
            # de modo que queda en las unidades de la curva y es comparable con
            # el ancho medio. `picp` y `mpiw` son los mismos numeros que
            # `cobertura_puntual` y `ancho_medio`, con el nombre del bloque:
            # se duplican para que una tabla del Bloque B se pueda leer sola.
            wn = pesos_normalizados(tau, pesos_tau)
            w_t = winkler(O, L, U, nivel=nivel) @ wn       # (n,)
            metricas["winkler"] = lambda idx, v=w_t: float(v[idx].mean())
            metricas["picp"] = metricas["cobertura_puntual"]
            metricas["mpiw"] = metricas["ancho_medio"]

    salida = ventana_movil(metricas, n, T0, w, paso, solapadas, t_offset,
                           verbose=verbose,
                           etiqueta=f"ventana_movil_funcional[w={w}]")
    salida.attrs["w"] = int(w)
    salida.attrs["solapadas"] = bool(solapadas)
    return salida
