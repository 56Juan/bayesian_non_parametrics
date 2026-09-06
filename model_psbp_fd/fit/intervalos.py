r"""
intervalos.py
=============
Intervalos de prediccion a un paso para modelos que solo entregan una
prediccion PUNTUAL, construidos desde el modelo de probabilidad que el propio
metodo supone.

De donde sale el intervalo, y por que esto NO es una banda inventada
--------------------------------------------------------------------
El FAR(1) de Bosq no es un algoritmo de prediccion puntual: es un MODELO,

    X_{t+1} = rho(X_t) + eps_{t+1},    eps iid, media cero, covarianza Sigma_eps,

de modo que la ley condicional de X_{t+1} dado X_t esta especificada por el
propio metodo y su intervalo de prediccion a un paso es el clasico del AR:
centrado en rho_hat(X_t) y con la dispersion de la innovacion. No hace falta
suponer nada que el modelo no suponga ya. Es la misma extension que en el AR
escalar lleva de `y_hat = phi_hat y_t` a `y_hat +/- z_{1-alpha/2} sigma_eps`.

Esto lo separa de una banda impuesta desde fuera --conformal, bootstrap de
residuos-- que no sale del modelo sino de un procedimiento de calibracion
adosado. Aqui la banda ES el modelo.

Las tres decisiones, y sus consecuencias
----------------------------------------
1. DISPERSION POR PUNTO DEL DOMINIO. `Sigma_eps` es un operador de covarianza,
   no un escalar: su diagonal sigma_eps(tau)^2 varia con tau. La banda usa esa
   diagonal (`por_tau=True`, el defecto) y por tanto NO tiene ancho constante:
   se ensancha donde la innovacion funcional es mas volatil. Con `por_tau=False`
   se colapsa a un unico sigma, que es mas estable pero borra esa estructura y
   deja una banda de ancho literalmente constante.

   Lo que la banda NO puede hacer, y hay que declararlo al reportar: ensancharse
   segun el ORIGEN. Sigma_eps es la misma para todo t, de modo que el ancho no
   responde al estado del proceso. En el Escenario 1 --homogeneo, gaussiano--
   eso es exactamente correcto y no hay nada que perder. En el Escenario 2, con
   volatilidad variable, es una limitacion real del competidor y no un defecto
   de esta implementacion: un FAR homocedastico no puede ensanchar en los
   brotes, y esa incapacidad es justamente lo que el PSBPM-FD deberia explotar.

2. FORMA GAUSSIANA. El cuantil es el de la normal. Bajo innovacion gaussiana
   --el Algoritmo 1-- es exacto; bajo innovacion asimetrica --el Algoritmo 4--
   la banda queda simetrica alrededor del centro y no puede representar el
   sesgo, lo que se vera como PIT inclinado y Winkler penalizado. Tambien eso
   es una propiedad del competidor, no un artefacto: el modelo lineal gaussiano
   no tiene forma que ofrecer. `cuantil` permite pasar el cuantil empirico de
   los residuos en su lugar cuando se quiera separar "falla por gaussiana" de
   "falla por homocedastica".

3. RESIDUOS DENTRO DE MUESTRA. `sigma_eps` se estima con los residuos del
   bloque de entrenamiento, que es donde el modelo se ajusto, de modo que
   subestima la dispersion: la banda sale algo mas angosta de lo que
   corresponde y su Winkler algo mejor de lo debido. La correccion de primer
   orden por grados de libertad (`ddof`) y la inflacion por incertidumbre de
   estimacion (`correccion_estimacion`) mitigan parte de eso, pero no lo
   eliminan. Es el sesgo que hay que declarar al comparar contra una banda
   bayesiana, que si integra la incertidumbre de los parametros.

Para que la comparacion sea limpia
----------------------------------
`residuos_para_banda` calcula los residuos contra EL MISMO objetivo con que se
evalua. En simulacion ese objetivo es la curva VERDADERA, no la observada, y
eso importa: estimando la dispersion contra los datos observados, sigma_eps
absorberia el ruido de medicion sigma_obs y la banda del competidor saldria
mas ancha por una razon que no tiene nada que ver con su calidad predictiva
--precisamente el ruido que el estudio decidio no atribuirle a nadie--. Con
datos reales no hay curva verdadera y el objetivo es la observada, de modo que
alli la banda si incluye sigma_obs, como debe.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "cuantil_normal",
    "sigma_residual",
    "residuos_para_banda",
    "banda_predictiva_modelo",
]


def cuantil_normal(p: float) -> float:
    """
    Cuantil de la normal estandar por biseccion sobre la funcion de error.

    Se calcula aqui y no con `scipy.stats.norm.ppf` para no introducir una
    dependencia de scipy en `fit/`, que hoy no la tiene: el modulo de metricas
    ya implementa su propia `erf` por la misma razon.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p={p} debe estar en (0, 1).")
    from math import erf, sqrt

    lo, hi = -40.0, 40.0
    for _ in range(200):
        med = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(med / sqrt(2.0))) < p:
            lo = med
        else:
            hi = med
    return 0.5 * (lo + hi)


def residuos_para_banda(X_objetivo: np.ndarray, X_pred: np.ndarray,
                        mascara_train: np.ndarray) -> np.ndarray:
    """
    Residuos del bloque de ENTRENAMIENTO, contra el objetivo de evaluacion.

    X_objetivo : (n, G) aquello contra lo que se mide el error. En simulacion,
        las curvas VERDADERAS; con datos reales, las observadas.
    X_pred     : (n, G) prediccion a h=1 del modelo, sobre la serie completa.
    mascara_train : (n,) booleana, True en los origenes de entrenamiento.

    Se toman SOLO los de entrenamiento: usar los de prueba para calibrar la
    banda con que despues se evalua el bloque de prueba es fuga, y ademas
    garantizaria una cobertura buena por construccion.
    """
    O = np.atleast_2d(np.asarray(X_objetivo, dtype=float))
    P = np.atleast_2d(np.asarray(X_pred, dtype=float))
    if O.shape != P.shape:
        raise ValueError(f"Formas incompatibles: {O.shape} vs {P.shape}.")
    m = np.asarray(mascara_train, dtype=bool).ravel()
    if m.size != O.shape[0]:
        raise ValueError(f"mascara_train tiene {m.size} entradas y hay "
                         f"{O.shape[0]} origenes.")
    if m.sum() < 2:
        raise ValueError("Se necesitan al menos 2 origenes de entrenamiento.")
    return O[m] - P[m]


def sigma_residual(residuos: np.ndarray, por_tau: bool = True,
                   ddof: int = 0) -> np.ndarray:
    """
    Desviacion de la innovacion estimada con los residuos de entrenamiento.

    residuos : (n_train, G).
    por_tau : True estima un sigma por punto del dominio --la diagonal de
        Sigma_eps-- y la banda hereda esa forma. False colapsa a un escalar
        (la raiz del promedio de la varianza sobre el dominio), que da una
        banda de ancho constante.
    ddof : grados de libertad que se descuentan. El defecto 0 es el estimador
        de maxima verosimilitud; `ddof = kn` descuenta las direcciones que el
        FAR estima, que es la correccion de primer orden razonable cuando
        n_train no es mucho mayor que kn. No se resta nada por defecto porque
        el numero de parametros efectivos depende del modelo y no puede
        adivinarse aqui.

    Los residuos NO se centran: bajo el modelo la innovacion tiene media cero,
    y centrarlos borraria un sesgo sistematico de la prediccion que la banda
    debe pagar. Si el modelo predice con sesgo, el intervalo tiene que quedar
    mal centrado, y eso es lo que el Winkler penaliza.
    """
    R = np.atleast_2d(np.asarray(residuos, dtype=float))
    n = R.shape[0]
    if n - ddof <= 0:
        raise ValueError(f"ddof={ddof} deja {n - ddof} grados de libertad.")
    var_tau = (R ** 2).sum(axis=0) / (n - ddof)          # (G,), sin centrar
    if por_tau:
        return np.sqrt(var_tau)
    return np.full(var_tau.size, float(np.sqrt(var_tau.mean())))


def banda_predictiva_modelo(X_pred: np.ndarray, residuos: np.ndarray,
                            nivel: float = 0.95, por_tau: bool = True,
                            ddof: int = 0,
                            correccion_estimacion: Optional[float] = None,
                            cuantil: Optional[float] = None):
    """
    Intervalo de prediccion a un paso implicado por el modelo. Retorna (li, ls).

    X_pred : (n, G) prediccion puntual sobre TODA la serie de origenes.
    residuos : (n_train, G) los del bloque de entrenamiento (`residuos_para_banda`).
    nivel : nivel nominal, p. ej. 0.95.
    por_tau, ddof : ver `sigma_residual`.
    correccion_estimacion : factor por el que se infla sigma para reconocer que
        rho esta ESTIMADO y no es conocido. En el AR escalar la varianza de
        prediccion a un paso es sigma^2 (1 + p/n) a primer orden, de modo que
        el valor natural es sqrt(1 + kn / n_train) con kn las direcciones
        retenidas. Es de primer orden y pequeño --con kn=6 y n_train=559 vale
        1.005--, de modo que omitirlo no cambia las conclusiones; se ofrece
        para poder declarar que se tuvo en cuenta.
    cuantil : sustituye al cuantil normal por uno dado, tipicamente el cuantil
        empirico de |residuo| / sigma. Sirve para separar "la banda falla
        porque supone gaussianidad" de "falla porque supone homocedasticidad".

    La banda es SIMETRICA alrededor de la prediccion y su ancho no depende del
    origen: son las dos propiedades del modelo lineal homogeneo, y las dos son
    lo que un modelo de mezcla puede hacer mejor. Al comparar Winkler contra
    una predictiva bayesiana hay que tenerlo presente: parte de la diferencia
    es de MODELO --que es lo que se quiere medir-- y parte es de MECANISMO de
    construccion del intervalo, y por eso conviene construir tambien la banda
    de este tipo para el modelo propuesto y mirar las tres filas juntas.
    """
    P = np.atleast_2d(np.asarray(X_pred, dtype=float))
    sd = sigma_residual(residuos, por_tau=por_tau, ddof=ddof)      # (G,)
    if sd.size != P.shape[1]:
        raise ValueError(f"Los residuos tienen {sd.size} puntos de grilla y "
                         f"X_pred {P.shape[1]}.")
    if correccion_estimacion is not None:
        if correccion_estimacion <= 0:
            raise ValueError("correccion_estimacion debe ser positiva.")
        sd = sd * float(correccion_estimacion)

    z = float(cuantil) if cuantil is not None else cuantil_normal(
        1.0 - (1.0 - nivel) / 2.0)
    ancho = z * sd[None, :]
    return P - ancho, P + ancho
