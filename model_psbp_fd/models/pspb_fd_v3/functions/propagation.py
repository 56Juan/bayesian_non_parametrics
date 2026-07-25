r"""
propagation.py — De la predictiva de los scores a la predictiva funcional
=========================================================================

El modelo propuesto ajusta un PSBPM univariado por cada score FPCA, bajo
independencia condicional entre scores dada la historia. La cantidad de interes
del estudio, sin embargo, no es el score sino la CURVA. Este modulo realiza el
transporte

    predictiva de los scores estandarizados
        -> des-estandarizacion
        -> reconstruccion FPCA
        -> predictiva funcional

propagando muestras y no solo momentos, que es lo unico que permite conservar
la forma de la mezcla a lo largo del transporte.

Por que hay que propagar MUESTRAS y no momentos
-----------------------------------------------
La reconstruccion FPCA es lineal en los scores,

    X(tau) = mu(tau) + sum_m xi_m psi_m(tau),

de modo que la media y la varianza puntual podrian obtenerse en forma cerrada.
Pero la mezcla de la predictiva de cada score induce, tras la combinacion
lineal, una predictiva funcional que en general NO es gaussiana ni unimodal, y
esa es justamente la propiedad que el estudio busca medir. Propagar solo dos
momentos la destruye. Ademas, el puntaje de energia y las bandas simultaneas
requieren extracciones conjuntas de la curva completa, no resumenes puntuales.

Independencia condicional entre scores
--------------------------------------
Las muestras de cada score se extraen de su propia cadena y se apilan. Esto es
exactamente lo que el modelo supone ---independencia condicional dada la
historia--- y no una aproximacion adicional introducida aqui. Conviene tener
presente, no obstante, que la dependencia contemporanea entre scores que el
modelo puede representar es solo la inducida por los rezagos comunes: cualquier
dependencia residual queda fuera por construccion, y una PIT funcional
sistematicamente sub-dispersa es su sintoma esperable.

El error de representacion NO es opcional
-----------------------------------------
La predictiva funcional obtenida por reconstruccion vive en el subespacio de M
autofunciones. La curva observada no: difiere de su proyeccion en el residuo de
truncamiento FPCA y en el error de suavizado de la base. Si ese residuo se
omite, las bandas funcionales son demasiado angostas por construccion y la
cobertura resultara inferior a la nominal por una razon que NADA tiene que ver
con el modelo dinamico. `modo_residuo` controla como incorporarlo:

    "ninguno"  : no se agrega. Solo apropiado si se evalua la curva PROYECTADA
                 y no la observada. Debe declararse explicitamente al reportar.
    "empirico" : se remuestrean curvas residuales completas del bloque de
                 entrenamiento. Preserva la estructura de correlacion del
                 residuo a lo largo del dominio, que es marcadamente no blanca.
                 Es la opcion recomendada.
    "puntual"  : ruido independiente por punto de la grilla con la varianza
                 puntual del residuo. Subestima la correlacion espacial y por
                 tanto ensancha de menos las bandas de funcionales agregados;
                 se ofrece como referencia.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "PropagadorFuncional",
    "residuos_representacion",
    "agrupar_muestras_cadenas",
    "bandas_puntuales",
]


# ==========================================================================
# UTILIDADES
# ==========================================================================

def residuos_representacion(X_obs: np.ndarray, X_rec: np.ndarray) -> np.ndarray:
    """
    Curvas residuales de la representacion sobre el bloque de ajuste.

    X_obs : (n, G) curvas observadas del bloque de ENTRENAMIENTO.
    X_rec : (n, G) su reconstruccion con M componentes FPCA.

    Deben provenir del bloque de entrenamiento: estimar el residuo con el
    bloque de prueba lo haria depender de aquello que se pretende evaluar.
    """
    O = np.atleast_2d(np.asarray(X_obs, dtype=float))
    R = np.atleast_2d(np.asarray(X_rec, dtype=float))
    if O.shape != R.shape:
        raise ValueError(f"Formas incompatibles: {O.shape} vs {R.shape}.")
    return O - R


def agrupar_muestras_cadenas(lista: Sequence[np.ndarray]) -> np.ndarray:
    """
    Concatena las extracciones de varias cadenas en una sola muestra.

    Es la representacion exacta de la mezcla de igual peso cuando las cadenas
    tienen la misma longitud tras el descarte inicial; con longitudes distintas
    la mezcla queda ponderada por el numero de extracciones y se advierte.
    """
    arrs = [np.asarray(a, dtype=float) for a in lista]
    if not arrs:
        raise ValueError("La lista de muestras esta vacia.")
    formas = {a.shape[1:] for a in arrs}
    if len(formas) > 1:
        raise ValueError(f"Dimensiones no iniciales heterogeneas: {formas}.")
    if len({a.shape[0] for a in arrs}) > 1:
        warnings.warn(
            "Cadenas de longitud distinta: la mezcla resultante queda "
            "ponderada por el numero de extracciones de cada una.",
            RuntimeWarning, stacklevel=2)
    return np.concatenate(arrs, axis=0)


def bandas_puntuales(muestras: np.ndarray, nivel: float = 0.95):
    """Cuantiles empiricos simetricos punto a punto; retorna (li, ls)."""
    Z = np.asarray(muestras, dtype=float)
    a = (1.0 - nivel) / 2.0
    return np.quantile(Z, a, axis=0), np.quantile(Z, 1.0 - a, axis=0)


# ==========================================================================
# PROPAGADOR
# ==========================================================================

class PropagadorFuncional:
    """
    Transporta la predictiva de los scores a la predictiva de las curvas.

    Parametros
    ----------
    Psi_grid : (G, M) autofunciones FPCA evaluadas en la grilla.
    mu_grid  : (G,)   funcion media del bloque de ajuste.
    estandarizador : objeto con `inverse_transform`, ajustado sobre los scores
        del bloque de entrenamiento. Si es None se asume que las muestras de
        scores ya vienen en escala original.
    residuos : (n_train, G), opcional. Curvas residuales de la representacion,
        obtenidas con `residuos_representacion` sobre el bloque de ajuste.
    modo_residuo : {"ninguno", "empirico", "puntual"}
    """

    def __init__(self, Psi_grid: np.ndarray, mu_grid: np.ndarray,
                 estandarizador=None, residuos: Optional[np.ndarray] = None,
                 modo_residuo: str = "empirico"):
        self.Psi = np.atleast_2d(np.asarray(Psi_grid, dtype=float))
        self.mu = np.asarray(mu_grid, dtype=float).ravel()
        if self.Psi.shape[0] != self.mu.size:
            raise ValueError(
                f"Psi_grid tiene {self.Psi.shape[0]} filas y mu_grid "
                f"{self.mu.size} puntos: no comparten grilla.")

        self.estandarizador = estandarizador
        self.modo_residuo = modo_residuo
        self.residuos = None
        self.sd_residuo = None

        if modo_residuo not in ("ninguno", "empirico", "puntual"):
            raise ValueError(
                f"modo_residuo='{modo_residuo}' invalido; use 'ninguno', "
                "'empirico' o 'puntual'.")
        if modo_residuo != "ninguno":
            if residuos is None:
                raise ValueError(
                    f"modo_residuo='{modo_residuo}' requiere `residuos`. Si la "
                    "intencion es evaluar la curva PROYECTADA y no la "
                    "observada, use modo_residuo='ninguno' de forma explicita.")
            R = np.atleast_2d(np.asarray(residuos, dtype=float))
            if R.shape[1] != self.mu.size:
                raise ValueError(
                    f"residuos tiene {R.shape[1]} columnas y la grilla "
                    f"{self.mu.size} puntos.")
            self.residuos = R
            self.sd_residuo = R.std(axis=0, ddof=1)

    # ------------------------------------------------------------------
    @property
    def M(self) -> int:
        return int(self.Psi.shape[1])

    @property
    def G(self) -> int:
        return int(self.Psi.shape[0])

    # ------------------------------------------------------------------
    def curvas_desde_scores(self, SCORES_STD: np.ndarray,
                            seed: Optional[int] = None) -> np.ndarray:
        """
        Convierte muestras de scores en muestras de curvas.

        SCORES_STD : (S, n, M) extracciones de la predictiva de los scores, en
                     la escala en que se entrenaron las cadenas.
        Retorna    : (S, n, G) extracciones de la predictiva funcional.
        """
        Z = np.asarray(SCORES_STD, dtype=float)
        if Z.ndim != 3:
            raise ValueError(f"SCORES_STD debe ser (S, n, M); recibido {Z.shape}.")
        S, n, M = Z.shape
        if M != self.M:
            raise ValueError(
                f"Las muestras tienen M={M} y las autofunciones M={self.M}.")

        plano = Z.reshape(S * n, M)
        if self.estandarizador is not None:
            plano = self.estandarizador.inverse_transform(plano)

        curvas = (self.mu[None, :] + plano @ self.Psi.T).reshape(S, n, self.G)

        if self.modo_residuo == "ninguno":
            return curvas

        rng = np.random.default_rng(seed)
        if self.modo_residuo == "empirico":
            idx = rng.integers(0, self.residuos.shape[0], size=(S, n))
            return curvas + self.residuos[idx]
        return curvas + rng.standard_normal((S, n, self.G)) * self.sd_residuo[None, None, :]

    # ------------------------------------------------------------------
    def momentos(self, SCORES_STD: np.ndarray,
                 seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Media y desviacion puntual de la predictiva funcional, mas la
        descomposicion del aporte del residuo de representacion.

        Se reporta `sd_sin_residuo` junto a `sd` porque la diferencia entre
        ambas cuantifica cuanta de la incertidumbre funcional proviene del
        truncamiento de la representacion y no del modelo dinamico: si el
        residuo domina, aumentar M rinde mas que mejorar el modelo.
        """
        X = self.curvas_desde_scores(SCORES_STD, seed=seed)
        salida = {"media": X.mean(axis=0), "sd": X.std(axis=0, ddof=1)}
        if self.modo_residuo != "ninguno":
            base = PropagadorFuncional(self.Psi, self.mu, self.estandarizador,
                                       modo_residuo="ninguno")
            salida["sd_sin_residuo"] = base.curvas_desde_scores(
                SCORES_STD).std(axis=0, ddof=1)
        return salida


# ==========================================================================
# ENSAMBLE DE MUESTRAS POR COMPONENTE Y CADENA
# ==========================================================================

def muestrear_scores(
    predictores: Dict[int, List],
    X_por_componente: Dict[int, np.ndarray],
    extracciones_por_iteracion: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Ensambla las muestras predictivas de todos los scores y cadenas.

    predictores : {m: [PSBPPredictor de cada cadena]} para m = 0..M-1.
    X_por_componente : {m: (n, p+1)} matriz de diseno con intercepto. Suele ser
        la misma para todos los scores ---las covariables son los rezagos de
        todas las componentes--- pero se acepta por componente para no imponer
        esa restriccion.

    Las cadenas de un mismo score se agrupan por concatenacion, que es la
    mezcla de igual peso; los scores se apilan bajo independencia condicional.
    Todas las cadenas deben aportar el mismo numero de extracciones, ya que de
    lo contrario el apilamiento entre scores emparejaria muestras de tamanos
    distintos y la dimension M no quedaria definida.

    Retorna (S, n, M).
    """
    claves = sorted(predictores)
    if claves != list(range(len(claves))):
        raise ValueError(
            f"Las claves de `predictores` deben ser 0..M-1; recibido {claves}.")

    ss = np.random.SeedSequence(seed)
    hijas = ss.spawn(len(claves))

    por_score = []
    for pos, m in enumerate(claves):
        cadenas = predictores[m]
        if not cadenas:
            raise ValueError(f"El score {m} no tiene cadenas asociadas.")
        Xm = X_por_componente[m]
        semillas = np.random.SeedSequence(hijas[pos].entropy,
                                          spawn_key=hijas[pos].spawn_key
                                          ).spawn(len(cadenas))
        muestras = [c.muestrear(Xm, extracciones_por_iteracion,
                                seed=int(s.generate_state(1)[0]))
                    for c, s in zip(cadenas, semillas)]
        por_score.append(agrupar_muestras_cadenas(muestras))

    tamanos = {a.shape for a in por_score}
    if len(tamanos) > 1:
        raise ValueError(
            f"Los scores producen muestras de formas distintas {tamanos}: "
            "verifique que todas las cadenas tengan igual longitud tras el "
            "descarte inicial y que el bloque evaluado sea el mismo.")
    return np.stack(por_score, axis=2)
