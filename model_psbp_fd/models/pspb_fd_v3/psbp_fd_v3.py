r"""
psbp_fd_v3.py — Orquestador de la inferencia predictiva funcional
=================================================================

Version 3: la clase ya no ajusta nada. El ajuste ocurre en MATLAB
(`psbp_train.m`), y esta capa consume las trazas resultantes para producir la
distribucion predictiva de las curvas, con su incertidumbre correctamente
propagada.

    trazas .mat por (score, cadena)
        -> PSBPPredictor          [predictiva de cada score]
        -> muestrear_scores       [agrupacion de cadenas + apilado de scores]
        -> PropagadorFuncional    [des-estandarizacion + reconstruccion FPCA]
        -> muestras de curvas (S, n, G)

Las muestras de curvas son el objeto que consume `fit`: CRPS muestral, puntaje
de energia, PIT, cobertura e intervalos se calculan todos a partir de ellas.

Cambio de contrato respecto de la v2
------------------------------------
En la v2, `predict(return_std=True)` devolvia la desviacion posterior de la
media condicional. Aqui devuelve la desviacion PREDICTIVA, que es la que
corresponde a una banda y a una medida de cobertura. La cantidad anterior sigue
disponible, pero hay que pedirla por su nombre (`sd_centro`), de modo que
usarla en lugar de la predictiva sea una decision explicita y no un accidente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .functions.predict import PSBPPredictor
from .functions.propagation import (
    PropagadorFuncional,
    bandas_puntuales,
    muestrear_scores,
    residuos_representacion,
)

__all__ = ["PSBP_FD_v3", "cargar_trazas_mat"]


# ==========================================================================
# LECTURA DE TRAZAS MATLAB
# ==========================================================================

_CLAVES = ("betajhout", "beta0hout", "tauhout", "alphahout",
           "psijhout", "Gammajhout", "gammajhout", "osumout", "inEout")


def cargar_trazas_mat(ruta) -> Dict[str, np.ndarray]:
    """
    Lee un archivo `chain_fpc_<idx>_iter<chain>.mat` producido por
    `psbp_train.m` y devuelve las trazas como arreglos de numpy.

    Se normaliza la forma de `beta0hout` y `tauhout`: MATLAB las guarda como
    (nsim, N, 1) o (nsim, N) segun la version, y el predictor exige (nsim, N).

    Nota sobre `inEout`: en `psbp_train.m` la cantidad se calcula ANTES de
    actualizar beta y tau dentro de la misma iteracion, mientras que las trazas
    de parametros se registran DESPUES. La consecuencia es un desfase de una
    iteracion entre `inEout[t]` y `betajhout[t]`, cuyo efecto sobre el promedio
    posterior es de orden 1/n_post. No invalida nada, pero explica por que la
    verificacion cruzada contra `predict` no da nunca exactamente cero y no debe
    perseguirse como si fuera un defecto.
    """
    from scipy.io import loadmat

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro la traza: {ruta}")
    crudo = loadmat(str(ruta), squeeze_me=False, struct_as_record=False)

    trazas: Dict[str, np.ndarray] = {}
    for k in _CLAVES:
        if k not in crudo:
            continue
        a = np.asarray(crudo[k], dtype=float)
        if k in ("beta0hout", "tauhout") and a.ndim == 3 and a.shape[2] == 1:
            a = a[:, :, 0]
        trazas[k] = a

    faltan = {"betajhout", "beta0hout", "tauhout", "alphahout",
              "psijhout", "Gammajhout"} - set(trazas)
    if faltan:
        raise KeyError(
            f"La traza {ruta.name} no contiene {sorted(faltan)}. Verifique que "
            "fue generada por psbp_train.m y no truncada al guardarse.")
    return trazas


# ==========================================================================
# ORQUESTADOR
# ==========================================================================

class PSBP_FD_v3:
    """
    Inferencia predictiva funcional a partir de trazas ya ajustadas.

    Parametros
    ----------
    predictores : {m: [PSBPPredictor por cadena]} con m = 0..M-1.
    propagador  : PropagadorFuncional ya construido.

    Uso tipico
    ----------
        modelo = PSBP_FD_v3.desde_artefactos(
            rutas_trazas={0: [...], 1: [...]},
            burn=500,
            Psi_grid=art.Psi_grid, mu_grid=art.mu_grid,
            estandarizador=std,
            residuos=residuos_representacion(X_train, X_train_rec),
        )
        Z   = modelo.muestrear_curvas(X_diseno, seed=0)   # (S, n, G)
        res = modelo.resumen(X_diseno, seed=0)
    """

    def __init__(self, predictores: Dict[int, List[PSBPPredictor]],
                 propagador: PropagadorFuncional):
        if not predictores:
            raise ValueError("`predictores` esta vacio.")
        claves = sorted(predictores)
        if claves != list(range(len(claves))):
            raise ValueError(
                f"Las claves deben ser 0..M-1; recibido {claves}.")
        if len(claves) != propagador.M:
            raise ValueError(
                f"Hay {len(claves)} scores con predictor y el propagador tiene "
                f"M={propagador.M} autofunciones: el truncamiento FPCA y el "
                "numero de modelos ajustados deben coincidir.")
        self.predictores = predictores
        self.propagador = propagador

    # ------------------------------------------------------------------
    @classmethod
    def desde_artefactos(
        cls,
        rutas_trazas: Dict[int, Sequence],
        burn: int,
        Psi_grid: np.ndarray,
        mu_grid: np.ndarray,
        estandarizador=None,
        residuos: Optional[np.ndarray] = None,
        modo_residuo: str = "empirico",
    ) -> "PSBP_FD_v3":
        """Construye el orquestador leyendo las trazas desde disco."""
        predictores = {
            m: [PSBPPredictor(cargar_trazas_mat(r), burn=burn) for r in rutas]
            for m, rutas in rutas_trazas.items()
        }
        prop = PropagadorFuncional(Psi_grid, mu_grid, estandarizador,
                                   residuos=residuos, modo_residuo=modo_residuo)
        return cls(predictores, prop)

    # ------------------------------------------------------------------
    @property
    def M(self) -> int:
        return len(self.predictores)

    def _diseno_por_componente(self, X) -> Dict[int, np.ndarray]:
        if isinstance(X, dict):
            return X
        return {m: X for m in self.predictores}

    # ------------------------------------------------------------------
    def muestrear_scores(self, X, extracciones_por_iteracion: int = 1,
                         seed: Optional[int] = None) -> np.ndarray:
        """Muestras de la predictiva de los scores, (S, n, M)."""
        return muestrear_scores(self.predictores,
                                self._diseno_por_componente(X),
                                extracciones_por_iteracion, seed)

    def muestrear_curvas(self, X, extracciones_por_iteracion: int = 1,
                         seed: Optional[int] = None) -> np.ndarray:
        """Muestras de la predictiva funcional, (S, n, G)."""
        Z = self.muestrear_scores(X, extracciones_por_iteracion, seed)
        return self.propagador.curvas_desde_scores(Z, seed=seed)

    # ------------------------------------------------------------------
    def momentos_scores(self, X) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Momentos predictivos por score, agrupando cadenas con la ley de
        varianza total: la varianza de la mezcla de cadenas es el promedio de
        las varianzas mas la varianza de las medias. Omitir el segundo termino
        subestima la incertidumbre justo cuando las cadenas mezclan mal, que es
        cuando mas importa detectarlo.
        """
        Xd = self._diseno_por_componente(X)
        salida = {}
        for m, cadenas in self.predictores.items():
            moms = [c.momentos_predictivos(Xd[m]) for c in cadenas]
            med = np.stack([d["media"] for d in moms], axis=1)      # (n, C)
            var = np.stack([d["var"] for d in moms], axis=1)
            mu = med.mean(axis=1)
            v = var.mean(axis=1) + ((med - mu[:, None]) ** 2).mean(axis=1)
            salida[m] = {
                "media": mu,
                "sd": np.sqrt(v),
                "sd_intra_cadena": np.sqrt(var.mean(axis=1)),
                "sd_entre_cadenas": np.sqrt(((med - mu[:, None]) ** 2).mean(axis=1)),
                "sd_centro": np.stack([d["sd_centro"] for d in moms], axis=1).mean(axis=1),
                "n_cadenas": len(cadenas),
            }
        return salida

    # ------------------------------------------------------------------
    def resumen(self, X, nivel: float = 0.95,
                extracciones_por_iteracion: int = 1,
                seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Prediccion funcional puntual, bandas y muestras en una sola pasada.

        Retorna un diccionario con:
            muestras : (S, n, G)
            media    : (n, G)
            sd       : (n, G) desviacion predictiva puntual
            li, ls   : (n, G) banda puntual de nivel `nivel`
            sd_sin_residuo : (n, G), presente si se incorporo el residuo de
                       representacion. Su distancia a `sd` cuantifica cuanta
                       incertidumbre proviene del truncamiento y no del modelo.

        Advertencia: `li` y `ls` son bandas PUNTUALES, no simultaneas. La
        probabilidad de que la curva completa quede contenida en todo el
        dominio es menor que el nivel nominal, por lo que reportarlas como
        cobertura funcional exige declararlo.
        """
        Z = self.muestrear_curvas(X, extracciones_por_iteracion, seed)
        li, ls = bandas_puntuales(Z, nivel)
        salida = {"muestras": Z, "media": Z.mean(axis=0),
                  "sd": Z.std(axis=0, ddof=1), "li": li, "ls": ls,
                  "nivel": nivel}
        if self.propagador.modo_residuo != "ninguno":
            base = PropagadorFuncional(self.propagador.Psi, self.propagador.mu,
                                       self.propagador.estandarizador,
                                       modo_residuo="ninguno")
            S = self.muestrear_scores(X, extracciones_por_iteracion, seed)
            salida["sd_sin_residuo"] = base.curvas_desde_scores(S).std(axis=0, ddof=1)
        return salida
