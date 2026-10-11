"""
artifacts.py
============
Contrato de persistencia entre los flujos del estudio.

El flujo de preprocesamiento escribe un conjunto de artefactos que el flujo de
resultados lee, y que MATLAB consume para el muestreo. Ese contrato involucra
mas de una decena de archivos y, mantenido a mano en cada notebook, es la
fuente de error mas persistente del proyecto: nombres que cambian en un lado y
no en el otro, matrices que `np.loadtxt` colapsa a una dimension cuando tienen
una sola columna, y verificaciones cruzadas que se omiten.

Este modulo concentra escritura y lectura en funciones emparejadas, de modo que
la definicion del contrato exista una sola vez. Las funciones de carga
garantizan la dimensionalidad de las matrices y las de verificacion cruzan los
metadatos entre artefactos antes de que el analisis comience.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "ArtefactosFPCA",
    "ArtefactosODPC",
    "guardar_curvas",
    "cargar_curvas",
    "cargar_curvas_true",
    "guardar_representacion",
    "cargar_representacion",
    "guardar_fpca",
    "cargar_fpca",
    "guardar_estandarizador",
    "cargar_estandarizador",
    "guardar_datasets_ar",
    "cargar_datasets_ar",
    "guardar_hiperparametros",
    "cargar_hiperparametros",
    "guardar_config_evaluacion",
    "cargar_config_evaluacion",
    "guardar_draws_scores",
    "cargar_draws_scores",
    "verificar_contrato",
]

# Nombres canonicos: unica definicion de la convencion de archivos.
ARCHIVOS = {
    "curvas":            "X_curves.npy",
    "curvas_true":       "X_curves_true.npy",
    "grilla":            "domain_grid.npy",
    "fr_pickle":         "functional_representation.pkl",
    "fr_config":         "fr_config.json",
    "theta":             "theta.csv",
    "basis_phi":         "basis_phi.csv",
    "fpca_eigenfun":     "fpca_eigenfunctions.csv",
    "fpca_mean":         "fpca_mean.csv",
    "fpca_gram_W":       "fpca_gram_W.csv",
    "fpca_coef_B":       "fpca_coef_B.csv",
    "fpca_mu_theta":     "fpca_mu_theta.csv",
    "fpca_scores":       "fpca_scores.csv",
    "fpca_scores_std":   "fpca_scores_std.csv",
    "fpca_evals":        "fpca_evals.csv",
    "fpca_meta":         "fpca_meta.json",
    # ODPC (solo corridas 110-112): filtros, base estatica y rezagos de la
    # reconstruccion
    "odpc_filtros":      "odpc_filtros.csv",
    "odpc_B_estatica":   "odpc_B_estatica.csv",
    "odpc_B_rezagos":    "odpc_B_rezagos.csv",
    "estandarizador":    "scores_standardizer",
    "manifest":          "datasets_manifest.json",
    "hiperparametros":   "hyperparameters.json",
    "eval_config":       "eval_config.json",
    # predict/: prediccion del PSBPM-FD que persiste el _04
    "banda_psbp":        "banda_funcional_psbp.npz",
    # .npy y no .npz: son cientos de MB por M y *.npy esta en .gitignore
    "draws_scores":      "draws_scores_psbp.npy",
    "comportamientos":   "comportamientos_psbp.npz",
}

# Matrices que deben conservar dos dimensiones aunque tengan una sola columna.
_MATRICES_2D = {"theta", "basis_phi", "fpca_eigenfun", "fpca_gram_W",
                "fpca_coef_B", "fpca_scores", "fpca_scores_std",
                "odpc_filtros", "odpc_B_estatica", "odpc_B_rezagos"}


# ==========================================================================
# UTILIDADES INTERNAS
# ==========================================================================

def _ruta(paths: dict, clave_dir: str, clave_archivo: str) -> Path:
    return Path(paths[clave_dir]) / ARCHIVOS[clave_archivo]


def _leer_matriz(path: Path, forzar_2d: bool) -> np.ndarray:
    """
    Lee un CSV numerico garantizando la dimensionalidad esperada.

    `np.loadtxt` colapsa a un arreglo unidimensional cuando el archivo tiene
    una sola columna, lo que rompe silenciosamente cualquier indexacion por
    columnas aguas abajo. Con una sola componente FPCA retenida ese caso deja
    de ser hipotetico.
    """
    arr = np.loadtxt(path, delimiter=",")
    if forzar_2d and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _escribir_json(obj: dict, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def _leer_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ==========================================================================
# CURVAS Y GRILLA
# ==========================================================================

def guardar_curvas(paths: dict, X: np.ndarray, grilla: np.ndarray,
                   X_true: Optional[np.ndarray] = None) -> dict:
    """
    Persiste las curvas del experimento y la grilla.

    X       : (T, G) curvas OBSERVADAS, con ruido de medicion. Es el unico
              objeto que puede alimentar la estimacion (base, FPCA,
              estandarizador, scores).
    X_true  : (T, G) curvas VERDADERAS del generador, sin ruido de medicion.
              Opcional porque los datos reales no la tienen. En simulacion es
              el objetivo de evaluacion: el error se mide contra la curva y no
              contra los datos, de modo que sigma_obs no contamina ambos lados
              de la comparacion. Nunca debe entrar a la estimacion.
    """
    d = Path(paths["functional"])
    np.save(d / ARCHIVOS["curvas"], np.asarray(X, dtype=float))
    np.save(d / ARCHIVOS["grilla"], np.asarray(grilla, dtype=float))
    salida = {"curvas": d / ARCHIVOS["curvas"], "grilla": d / ARCHIVOS["grilla"]}

    if X_true is not None:
        Xt = np.atleast_2d(np.asarray(X_true, dtype=float))
        if Xt.shape != np.atleast_2d(np.asarray(X, dtype=float)).shape:
            raise ValueError(
                f"X_true tiene forma {Xt.shape} y X {np.shape(X)}: deben "
                "provenir de la misma replica y la misma grilla.")
        np.save(d / ARCHIVOS["curvas_true"], Xt)
        salida["curvas_true"] = d / ARCHIVOS["curvas_true"]
    return salida


def cargar_curvas(paths: dict) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (X, grilla). X es (T, G) OBSERVADA, en escala original."""
    d = Path(paths["functional"])
    for clave in ("curvas", "grilla"):
        p = d / ARCHIVOS[clave]
        if not p.exists():
            raise FileNotFoundError(
                f"No se encontro {p}. Ejecute primero el flujo de preprocesamiento."
            )
    X = np.load(d / ARCHIVOS["curvas"])
    tau = np.load(d / ARCHIVOS["grilla"])
    return np.atleast_2d(X), np.asarray(tau, dtype=float).ravel()


def cargar_curvas_true(paths: dict, estricto: bool = True) -> Optional[np.ndarray]:
    """
    Retorna (T, G) las curvas VERDADERAS sin ruido, o None si no se persistieron.

    Con `estricto=True` la ausencia es un error: en un experimento de
    simulacion evaluar contra los datos observados en lugar de contra la curva
    infla el error y la cobertura por una razon ajena al modelo, y ese sesgo no
    es visible en los numeros resultantes.
    """
    p = Path(paths["functional"]) / ARCHIVOS["curvas_true"]
    if not p.exists():
        if estricto:
            raise FileNotFoundError(
                f"No se encontro {p}. El flujo de preprocesamiento debe llamar "
                "guardar_curvas(..., X_true=salida.curvas[REPLICA_IDX]).")
        return None
    return np.atleast_2d(np.load(p))


# ==========================================================================
# REPRESENTACION FUNCIONAL
# ==========================================================================

def guardar_representacion(paths: dict, fr, THETA: np.ndarray,
                           extra: Optional[dict] = None) -> dict:
    """Serializa la representacion ajustada y su configuracion legible."""
    d = Path(paths["functional"])
    with open(d / ARCHIVOS["fr_pickle"], "wb") as f:
        pickle.dump(fr, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.savetxt(d / ARCHIVOS["theta"], np.atleast_2d(THETA), delimiter=",")

    cfg = {
        "method": getattr(fr, "method", "bspline"),
        "n_basis": int(np.atleast_2d(THETA).shape[1]),
        "order": getattr(fr, "order", None),
        "T": int(np.atleast_2d(THETA).shape[0]),
        "K": int(np.atleast_2d(THETA).shape[1]),
    }
    if extra:
        cfg.update(extra)
    _escribir_json(cfg, d / ARCHIVOS["fr_config"])
    return {"fr": d / ARCHIVOS["fr_pickle"], "theta": d / ARCHIVOS["theta"],
            "config": d / ARCHIVOS["fr_config"]}


def cargar_representacion(paths: dict, estricto: bool = True):
    """
    Retorna (fr, THETA, fr_config).

    Con `estricto=False` y pickle ausente retorna fr=None en lugar de fallar,
    util para inspeccionar artefactos sin el entorno completo. Notese que en
    ese caso las secciones del analisis que dependen de `fr` se omitiran de
    forma silenciosa, por lo que `estricto=True` es el valor recomendado.
    """
    d = Path(paths["functional"])
    p_fr = d / ARCHIVOS["fr_pickle"]
    if p_fr.exists():
        with open(p_fr, "rb") as f:
            fr = pickle.load(f)
    elif estricto:
        raise FileNotFoundError(f"No se encontro {p_fr}.")
    else:
        fr = None

    THETA = _leer_matriz(d / ARCHIVOS["theta"], True)
    cfg = _leer_json(d / ARCHIVOS["fr_config"]) if (d / ARCHIVOS["fr_config"]).exists() else {}
    return fr, THETA, cfg


# ==========================================================================
# FPCA
# ==========================================================================

@dataclass
class ArtefactosFPCA:
    """
    Objetos de la descomposicion FPCA leidos desde disco.

    Psi_grid : (G, M) autofunciones en la grilla
    mu_grid  : (G,)   media funcional en la grilla
    W        : (K, K) matriz de Gram en metrica L^2
    B        : (K, M) coeficientes de las autofunciones
    mu_theta : (K,)   media de los coeficientes
    Phi      : (G, K) base evaluada en la grilla
    SCORES   : (T, M) scores en escala original
    SCORES_STD : (T, M) scores estandarizados, si fueron persistidos; None en
               caso contrario. Se declara explicitamente porque `guardar_fpca`
               los escribe de forma opcional: sin este campo el artefacto se
               producia en disco pero no se recuperaba, y un consumidor que
               esperara scores estandarizados obtenia silenciosamente los de
               escala original ---confusion de escalas sin error visible, y
               justo en la frontera que el esquema de retencion temporal exige
               mantener nitida.
    evals    : (K,)   autovalores
    meta     : metadatos (M, K, var_explicada, ...)
    """

    Psi_grid: np.ndarray
    mu_grid: np.ndarray
    W: np.ndarray
    B: np.ndarray
    mu_theta: np.ndarray
    Phi: np.ndarray
    SCORES: np.ndarray
    SCORES_STD: Optional[np.ndarray] = None
    evals: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def M(self) -> int:
        return int(self.Psi_grid.shape[1])

    @property
    def K(self) -> int:
        return int(self.Phi.shape[1])

    def reconstruct(self, SCORES: np.ndarray) -> np.ndarray:
        """Scores en escala original -> curvas en la grilla."""
        S = np.atleast_2d(np.asarray(SCORES, dtype=float))
        return self.mu_grid[None, :] + S @ self.Psi_grid.T

    def scores_a_theta(self, SCORES: np.ndarray) -> np.ndarray:
        """Scores en escala original -> coeficientes de la base."""
        S = np.atleast_2d(np.asarray(SCORES, dtype=float))
        return S @ self.B.T + self.mu_theta[None, :]

    def transform(self, THETA: np.ndarray) -> np.ndarray:
        """
        Coeficientes de la base -> scores (T, M), en escala original.

        Replica `FPCA_L2.transform` sobre los artefactos leidos desde disco:

            xi = (theta - mu_theta) W B,

        donde `mu_theta`, `W` y `B` fueron estimados exclusivamente con el
        bloque de entrenamiento. Por construccion, proyectar coeficientes de
        prueba con este metodo no introduce fuga de informacion: el centrado
        emplea siempre la media del bloque de ajuste.

        Parametros
        ----------
        THETA : (T, K) coeficientes de las curvas en la base B-spline
                (salida de `FunctionalRepresentation.transform`).

        Retorna
        -------
        SCORES : (T, M) scores FPCA en escala original (NO estandarizados;
                 aplicar despues el `DataStandardizer` persistido si el
                 modelo consume scores estandarizados).
        """
        TH = np.atleast_2d(np.asarray(THETA, dtype=float))
        if TH.shape[1] != self.K:
            raise ValueError(
                f"THETA tiene {TH.shape[1]} columnas y la base persistida "
                f"tiene K={self.K}: no corresponden al mismo sistema."
            )
        return (TH - self.mu_theta[None, :]) @ (self.W @ self.B)

    def verificar(self, tol: float = 1e-10) -> dict:
        """
        Ortonormalidad y equivalencia de las dos rutas de reconstruccion.

        La tolerancia por defecto se fijo en 1e-10 y no en 1e-8. Con la FPCA
        generalizada en metrica L^2 el error de ortonormalidad es de orden
        1e-14 a 1e-13 incluso con matrices de Gram mal condicionadas, de modo
        que una tolerancia de 1e-8 admitiria una regresion de cuatro ordenes de
        magnitud sin senalarla. El margen que queda sigue siendo amplio
        respecto del piso numerico real.
        """
        err_orto = float(np.abs(self.B.T @ self.W @ self.B - np.eye(self.M)).max())
        S = np.eye(self.M)
        err_rutas = float(np.abs(
            self.reconstruct(S) - self.scores_a_theta(S) @ self.Phi.T).max())
        return {"err_ortonormalidad_gram": err_orto,
                "err_rutas_reconstruccion": err_rutas,
                "todo_ok": bool(max(err_orto, err_rutas) < tol)}


@dataclass
class ArtefactosODPC(ArtefactosFPCA):
    """
    Artefactos de `ODPC_Funcional`. `Psi_grid` / `B` / `reconstruct` son la
    parte CONTEMPORANEA de la reconstruccion (la que depende del score que se
    predice), de modo que PropagadorFuncional y las lineas base funcionan igual
    que con la FPCA estatica. La parte de los rezagos f_{t-1..t-k2}, conocida en
    el origen, sale de `desplazamiento` y se SUMA a la curva.

    - `transform` filtra la serie CONTINUA: la fila t usa las curvas t-k1..t.
    - `B` no es ortonormal en W; se verifica la base estatica de partida.

    filtros    : (k1+1, K_din, M)  filtros[j] pesa la curva t-j.
    B_estatica : (K, K_din)        base FPCA estatica ortonormal de partida.
    B_rezagos  : (k2, K, M)        coeficientes de base de f_{t-h}, h = 1..k2.
    """

    filtros: np.ndarray = None
    B_estatica: np.ndarray = None
    B_rezagos: np.ndarray = None

    def scores_estaticos(self, THETA: np.ndarray) -> np.ndarray:
        TH = np.atleast_2d(np.asarray(THETA, dtype=float))
        if TH.shape[1] != self.K:
            raise ValueError(f"THETA tiene {TH.shape[1]} columnas y K={self.K}.")
        return (TH - self.mu_theta[None, :]) @ (self.W @ self.B_estatica)

    def transform(self, THETA: np.ndarray) -> np.ndarray:
        """Serie CONTINUA de coeficientes (T, K) -> componentes (T, M)."""
        from ..functions_models.functions_odpc import filtrar_causal
        return filtrar_causal(self.scores_estaticos(THETA), self.filtros)

    def transform_prediccion(self, THETA: np.ndarray, THETA_pred: np.ndarray,
                             desde: int) -> np.ndarray:
        """
        Componentes con la curva t sustituida por su prediccion a un paso,
        t = desde..T-1, y rezagos reales: f_t + a_0^T (s_pred_t - s_t).
        """
        Y = self.transform(THETA)[desde:]
        S_real = self.scores_estaticos(THETA)[desde:]
        S_pred = self.scores_estaticos(THETA_pred)
        if S_pred.shape != S_real.shape:
            raise ValueError(f"THETA_pred {S_pred.shape} no alinea con "
                             f"THETA[{desde}:] {S_real.shape}.")
        return Y + (S_pred - S_real) @ self.filtros[0]

    def desplazamiento(self, SCORES: Optional[np.ndarray] = None,
                       desde: int = 0) -> np.ndarray:
        """
        (T-desde, G) aporte de f_{t-1..t-k2} a la curva de t. Por defecto con
        la serie persistida: son scores REALES, conocidos en el origen.
        """
        from ..functions_models.functions_odpc import rezagar
        F = self.SCORES if SCORES is None else np.atleast_2d(SCORES)
        D = np.zeros((F.shape[0], self.Phi.shape[0]))
        for h in range(1, self.B_rezagos.shape[0] + 1):
            D += rezagar(F, h) @ (self.Phi @ self.B_rezagos[h - 1]).T
        return D[desde:]

    def reconstruct_serie(self, SCORES: Optional[np.ndarray] = None) -> np.ndarray:
        """Serie CONTINUA de componentes -> curvas (T, G), todos los rezagos."""
        F = self.SCORES if SCORES is None else np.atleast_2d(SCORES)
        return self.reconstruct(F) + self.desplazamiento(F)

    def verificar(self, tol: float = 1e-10) -> dict:
        Kd = self.B_estatica.shape[1]
        err_orto = float(np.abs(
            self.B_estatica.T @ self.W @ self.B_estatica - np.eye(Kd)).max())
        S = np.eye(self.M)
        err_rutas = float(np.abs(
            self.reconstruct(S) - self.scores_a_theta(S) @ self.Phi.T).max())
        err_forma = float(self.filtros.shape[1:] != (Kd, self.M)
                          or self.B_rezagos.shape[1:] != (self.K, self.M))
        return {"err_ortonormalidad_estatica": err_orto,
                "err_rutas_reconstruccion": err_rutas,
                "err_forma": err_forma,
                "todo_ok": bool(max(err_orto, err_rutas, err_forma) < tol)}


def guardar_fpca(paths: dict, fpca, SCORES: np.ndarray,
                 SCORES_STD: Optional[np.ndarray] = None,
                 meta_extra: Optional[dict] = None) -> dict:
    """
    Persiste la descomposicion FPCA.

    `fpca` es una instancia de `FPCA_L2` ya ajustada y con M fijado. Los scores
    se pasan por separado porque corresponden a la serie completa, mientras que
    el ajuste se realizo unicamente con el bloque de entrenamiento.
    """
    d = Path(paths["functional"])
    tablas = {
        "fpca_eigenfun": fpca.Psi_grid,
        "fpca_mean": fpca.mu_grid,
        "fpca_gram_W": fpca.W,
        "fpca_coef_B": fpca.B,
        "fpca_mu_theta": fpca.mu_theta,
        "basis_phi": fpca.Phi,
        "fpca_scores": np.atleast_2d(SCORES),
        "fpca_evals": fpca.evals,
    }
    odpc = hasattr(fpca, "betas_full")
    if odpc:
        F = fpca.filtros                                      # (k1+1, Kd, M)
        tablas["odpc_filtros"] = F.reshape(-1, F.shape[2])
        tablas["odpc_B_estatica"] = fpca.B_full[:, :F.shape[1]]
        # (k2*K, M): bloques B_h apilados, h = 1..k2
        if int(fpca.k2) > 0:
            tablas["odpc_B_rezagos"] = np.vstack(
                [fpca.B_rezago(h) for h in range(1, int(fpca.k2) + 1)])
    if SCORES_STD is not None:
        tablas["fpca_scores_std"] = np.atleast_2d(SCORES_STD)
    else:
        # Sin esta limpieza, un archivo de una corrida anterior sobrevive al
        # reescribir la FPCA y `cargar_fpca` lo devolveria emparejado con
        # scores nuevos, posiblemente con otro M. La inconsistencia no produce
        # error: solo resultados en la escala equivocada.
        obsoleto = d / ARCHIVOS["fpca_scores_std"]
        if obsoleto.exists():
            obsoleto.unlink()

    for clave, arr in tablas.items():
        np.savetxt(d / ARCHIVOS[clave], np.asarray(arr, dtype=float), delimiter=",")

    meta = {
        "M": int(fpca.M),
        "K": int(fpca.evals.size),
        "var_explained": float(fpca.var_cum[fpca.M - 1]),
        "n_ajuste": int(fpca.n_ajuste),
        # Condicionamiento de la matriz de Gram. Se persiste porque la raiz
        # inversa W^{-1/2} empleada en el ajuste amplifica el ruido de las
        # direcciones asociadas a los autovalores menores de W; sin este
        # registro, comparar escenarios con distinto K no permite distinguir
        # una diferencia sustantiva de un artefacto numerico.
        "cond_W": (float(fpca.cond_W)
                   if getattr(fpca, "cond_W", None) is not None else None),
        "lambdas": [float(x) for x in fpca.lambdas],
    }
    if odpc:
        # var_explained: varianza de train que recupera la reconstruccion con
        # todos los rezagos (la que usa la evaluacion).
        meta.update({
            "tipo": "odpc",
            "K": int(fpca.B_full.shape[0]),     # evals tiene M_max, no K
            "var_explained": float(fpca.var_reconstruida()),
            "var_estatica": float(fpca.var_cum_estatica[fpca.M - 1]),
            "k1": int(fpca.k1), "k2": int(fpca.k2),
            "K_din": int(fpca.filtros_full.shape[1]),
            "iteraciones_als": [int(x) for x in fpca.iteraciones[:fpca.M]],
        })
    else:
        meta["tipo"] = "estatica"
        for clave in ("odpc_filtros", "odpc_B_estatica", "odpc_B_rezagos"):
            obsoleto = d / ARCHIVOS[clave]
            if obsoleto.exists():
                obsoleto.unlink()
    if meta_extra:
        meta.update(meta_extra)
    _escribir_json(meta, d / ARCHIVOS["fpca_meta"])
    return {"dir": d, "meta": meta}


def cargar_fpca(paths: dict) -> ArtefactosFPCA:
    """Carga la FPCA garantizando que toda matriz conserve dos dimensiones."""
    d = Path(paths["functional"])
    faltantes = [ARCHIVOS[c] for c in
                 ("fpca_eigenfun", "fpca_mean", "fpca_gram_W", "fpca_coef_B",
                  "fpca_mu_theta", "basis_phi", "fpca_scores")
                 if not (d / ARCHIVOS[c]).exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan artefactos FPCA en {d}: {faltantes}. Ejecute primero el "
            "flujo de preprocesamiento."
        )

    def leer(clave):
        return _leer_matriz(d / ARCHIVOS[clave], clave in _MATRICES_2D)

    evals_p = d / ARCHIVOS["fpca_evals"]
    meta_p = d / ARCHIVOS["fpca_meta"]
    scores_std_p = d / ARCHIVOS["fpca_scores_std"]
    meta = _leer_json(meta_p) if meta_p.exists() else {}
    extra, clase = {}, ArtefactosFPCA
    if meta.get("tipo") == "odpc":
        F = leer("odpc_filtros")
        k1, k2, Kd = int(meta["k1"]), int(meta["k2"]), int(meta["K_din"])
        B_est = leer("odpc_B_estatica")
        M_, K_ = F.shape[1], B_est.shape[0]
        extra = {"filtros": F.reshape(k1 + 1, Kd, M_),
                 "B_estatica": B_est,
                 "B_rezagos": (leer("odpc_B_rezagos").reshape(k2, K_, M_)
                               if k2 > 0 else np.zeros((0, K_, M_)))}
        clase = ArtefactosODPC
    return clase(**extra,
        Psi_grid=leer("fpca_eigenfun"),
        mu_grid=np.ravel(leer("fpca_mean")),
        W=leer("fpca_gram_W"),
        B=leer("fpca_coef_B"),
        mu_theta=np.ravel(leer("fpca_mu_theta")),
        Phi=leer("basis_phi"),
        SCORES=leer("fpca_scores"),
        SCORES_STD=(_leer_matriz(scores_std_p, True)
                    if scores_std_p.exists() else None),
        evals=np.ravel(np.loadtxt(evals_p, delimiter=",")) if evals_p.exists() else None,
        meta=meta,
    )


# ==========================================================================
# ESTANDARIZADOR
# ==========================================================================

def guardar_estandarizador(paths: dict, estandarizador) -> Path:
    """Persiste el estandarizador de scores en su propio subdirectorio."""
    p = Path(paths["functional"]) / ARCHIVOS["estandarizador"]
    estandarizador.save(p)
    return p


def cargar_estandarizador(paths: dict, clase):
    """
    Carga el estandarizador de scores.

    `clase` es la clase con el metodo de clase `load` (DataStandardizer); se
    recibe como argumento para no acoplar este modulo de persistencia a la
    implementacion concreta del preprocesamiento.
    """
    p = Path(paths["functional"]) / ARCHIVOS["estandarizador"]
    if not (p / "standardizer_metadata.json").exists():
        raise FileNotFoundError(
            f"No se encontro el estandarizador en {p}. Sin el, las predicciones "
            "no pueden des-estandarizarse y la reconstruccion funcional queda "
            "en unidades incorrectas."
        )
    return clase.load(p)


# ==========================================================================
# DATASETS AR(p) Y MANIFEST
# ==========================================================================

def nombre_dataset(fpc_idx: int, bloque: str) -> str:
    """Convencion de nombres: dataset_fpc_<idx base 1>_<bloque>.csv."""
    if bloque not in ("train", "test"):
        raise ValueError(f"bloque debe ser 'train' o 'test'; recibido {bloque!r}.")
    return f"dataset_fpc_{int(fpc_idx)}_{bloque}.csv"


def guardar_datasets_ar(paths: dict, dfs_train: dict, dfs_test: dict,
                        manifest: dict) -> dict:
    """Escribe los datasets por bloque y el manifest que los describe."""
    d = Path(paths["functional"])
    component_idx = manifest["component_idx"]
    manifest = dict(manifest)
    manifest["datasets"] = {"train": {}, "test": {}}

    for bloque, dfs in (("train", dfs_train), ("test", dfs_test)):
        for k, tabla in dfs.items():
            fpc_idx = int(component_idx[k]) + 1
            fname = nombre_dataset(fpc_idx, bloque)
            tabla.to_csv(d / fname, index=False)
            manifest["datasets"][bloque][fname] = {
                "response": f"fpc_{fpc_idx}", "shape": list(tabla.shape)}

    _escribir_json(manifest, d / ARCHIVOS["manifest"])
    return {"dir": d, "manifest": manifest}


def cargar_datasets_ar(paths: dict, bloque: str = "train") -> tuple[dict, dict]:
    """
    Carga los datasets de un bloque y el manifest.

    Retorna (dfs, manifest) con dfs[k] indexado por componente del modelo.
    """
    d = Path(paths["functional"])
    p_man = d / ARCHIVOS["manifest"]
    if not p_man.exists():
        raise FileNotFoundError(f"No se encontro {p_man}.")
    manifest = _leer_json(p_man)

    component_idx = manifest["component_idx"]
    dfs, faltantes = {}, []
    for k, idx in enumerate(component_idx):
        fname = nombre_dataset(int(idx) + 1, bloque)
        p = d / fname
        if p.exists():
            dfs[k] = pd.read_csv(p)
        else:
            faltantes.append(fname)
    if faltantes:
        raise FileNotFoundError(
            f"Faltan datasets del bloque '{bloque}' en {d}: {faltantes}."
        )
    return dfs, manifest


# ==========================================================================
# HIPERPARAMETROS Y CONFIGURACION DE EVALUACION
# ==========================================================================

def guardar_hiperparametros(paths: dict, hp_artifact: dict) -> Path:
    """Persiste el contrato de hiperparametros que consume MATLAB."""
    return _escribir_json(hp_artifact,
                          Path(paths["out_artefact"]) / ARCHIVOS["hiperparametros"])


def cargar_hiperparametros(paths: dict, verificar: bool = True) -> dict:
    """Carga los hiperparametros y verifica que contengan las claves del contrato."""
    p = Path(paths["out_artefact"]) / ARCHIVOS["hiperparametros"]
    if not p.exists():
        raise FileNotFoundError(f"No se encontro {p}.")
    hp = _leer_json(p)
    if verificar:
        requeridas = ("n_iter", "mcmc_config", "seed_base", "scores_scale",
                      "hyperparams_list")
        faltan = [c for c in requeridas if c not in hp]
        if faltan:
            raise KeyError(
                f"hyperparameters.json incompleto, faltan {faltan}. Regenere "
                "con el flujo de preprocesamiento actualizado."
            )
    return hp


def guardar_config_evaluacion(paths: dict, eval_config: dict) -> Path:
    """Persiste la configuracion del esquema de evaluacion."""
    return _escribir_json(eval_config,
                          Path(paths["out_artefact"]) / ARCHIVOS["eval_config"])


def cargar_config_evaluacion(paths: dict) -> dict:
    p = Path(paths["out_artefact"]) / ARCHIVOS["eval_config"]
    if not p.exists():
        raise FileNotFoundError(f"No se encontro {p}.")
    return _leer_json(p)


# ==========================================================================
# EXTRACCIONES DE LA PREDICTIVA (predict/)
# ==========================================================================

def guardar_draws_scores(paths: dict, SC_draws: np.ndarray) -> Path:
    """Extracciones de scores del PSBPM-FD, (S, n, M) en float32."""
    p = Path(paths["predict"]) / ARCHIVOS["draws_scores"]
    np.save(p, np.asarray(SC_draws, dtype=np.float32))
    return p


def cargar_draws_scores(paths: dict, mmap: bool = False) -> Optional[np.ndarray]:
    """(S, n, M) o None si no se persistieron."""
    p = Path(paths["predict"]) / ARCHIVOS["draws_scores"]
    if not p.exists():
        return None
    return np.load(p, mmap_mode="r" if mmap else None)


# ==========================================================================
# VERIFICACION CRUZADA DEL CONTRATO
# ==========================================================================

def verificar_contrato(paths: dict, estricto: bool = True) -> dict:
    """
    Cruza los metadatos de los artefactos antes de iniciar el analisis.

    Comprueba que el manifest y los hiperparametros declaren el mismo numero de
    componentes y la misma escala de scores, que la particion temporal este
    registrada, y que las dimensiones de los artefactos FPCA sean mutuamente
    consistentes. Cada una de estas discrepancias, de no detectarse aqui,
    produce mas adelante un error criptico o, peor, un resultado incorrecto sin
    error alguno.
    """
    informe, problemas = {}, []

    manifest = _leer_json(Path(paths["functional"]) / ARCHIVOS["manifest"])
    hp = cargar_hiperparametros(paths, verificar=False)
    fpca = cargar_fpca(paths)

    n_man = int(manifest["n_components"])
    n_hp = len(hp.get("hyperparams_list", []))
    if n_man != n_hp:
        problemas.append(f"n_components discrepa: manifest={n_man}, hiperparametros={n_hp}")

    if manifest.get("scores_scale") != hp.get("scores_scale"):
        problemas.append(
            f"scores_scale discrepa: manifest={manifest.get('scores_scale')!r}, "
            f"hiperparametros={hp.get('scores_scale')!r}")

    if "T0" not in manifest:
        problemas.append("manifest sin 'T0': la particion temporal no esta registrada")

    M_meta = int(fpca.meta.get("M", fpca.M))
    if M_meta != fpca.M:
        problemas.append(f"M discrepa: meta={M_meta}, autofunciones={fpca.M}")
    if fpca.B.shape != (fpca.K, fpca.M):
        problemas.append(f"B tiene forma {fpca.B.shape}, se esperaba ({fpca.K}, {fpca.M})")
    if fpca.SCORES.shape[1] != fpca.M:
        problemas.append(
            f"SCORES tiene {fpca.SCORES.shape[1]} columnas y M={fpca.M}")

    ver = fpca.verificar()
    if not ver["todo_ok"]:
        problemas.append(f"identidades FPCA no se cumplen: {ver}")

    # Retencion temporal del estandarizador. Se lee el metadato directamente y
    # no via `cargar_estandarizador` para no acoplar este modulo a la clase de
    # preprocesamiento. La comprobacion existe porque la correccion del estudio
    # depende de que los momentos de estandarizacion provengan solo del bloque
    # de entrenamiento, y ese hecho no es observable en los scores resultantes:
    # un ajuste sobre la serie completa produce numeros perfectamente
    # plausibles y resultados fuera de muestra invalidos.
    T0_man = manifest.get("T0")
    meta_std = Path(paths["functional"]) / ARCHIVOS["estandarizador"] / \
        "standardizer_metadata.json"
    if meta_std.exists() and T0_man is not None:
        n_aj = _leer_json(meta_std).get("n_ajuste")
        informe["estandarizador_n_ajuste"] = n_aj
        if n_aj is None:
            problemas.append(
                "el estandarizador no registra 'n_ajuste' (artefacto anterior "
                "al registro del bloque de ajuste): la retencion temporal no "
                "puede verificarse y debe regenerarse")
        elif int(n_aj) != int(T0_man):
            problemas.append(
                f"el estandarizador se ajusto con {n_aj} filas y T0={T0_man}: "
                "los momentos de estandarizacion vieron el bloque de prueba")

    # Las exogenas (no scores) se declaran en el manifest y no cuentan como rezagos.
    exo_man = list(manifest.get("exogenas", {}))
    p_prev = len([c for c in manifest.get("cov_names", []) if c not in exo_man])
    p_esp = n_man * int(manifest.get("n_lags", 0))
    if any(not c.startswith("fpc_") for c in manifest.get("cov_names", []) if c not in exo_man):
        problemas.append("cov_names trae covariables que no son scores y no estan en 'exogenas'")
    if p_prev != p_esp:
        problemas.append(f"cov_names tiene {p_prev} entradas, se esperaban {p_esp}")

    informe.update({
        "n_components": n_man, "M": fpca.M, "K": fpca.K,
        "n_lags": manifest.get("n_lags"),
        "T": manifest.get("T"), "T0": manifest.get("T0"),
        "scores_scale": manifest.get("scores_scale"),
        "n_iter": hp.get("n_iter"), "mcmc_config": hp.get("mcmc_config"),
        "verificacion_fpca": ver,
        "problemas": problemas,
        "contrato_ok": not problemas,
    })

    if problemas and estricto:
        raise ValueError("Contrato de artefactos inconsistente:\n  - "
                         + "\n  - ".join(problemas))
    return informe