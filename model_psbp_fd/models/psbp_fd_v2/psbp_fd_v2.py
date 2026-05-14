"""
PSBP_FD_v1 — Clase orquestadora del modelo Probit Stick-Breaking Process
=========================================================================

Compone los componentes funcionales del paquete:

    pd.DataFrame (estandarizado externamente) ──► PSBPSampler ──► PSBPPredictor

CAMBIOS v2 (D3 — opción A):
---------------------------
La clase ya **NO estandariza ni desestandariza**. Recibe datos que el
usuario debe haber estandarizado externamente. Esto elimina la
inconsistencia MATLAB (estandarizar con ddof=0, desestandarizar con
ddof=1) y simplifica radicalmente el contrato.

El usuario es responsable de:
    1. Estandarizar X y y antes de llamar a `fit`.
    2. Estandarizar X usando las MISMAS estadísticas del train antes
       de llamar a `predict`.
    3. Desestandarizar las predicciones si las necesita en escala
       original.

Author: model_psbp_fd
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

from .functions.sampler import PSBPSampler
from .functions.predict import PSBPPredictor


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class PSBP_FD_v1:
    """
    Modelo PSBP-FD versión 1 — orquestador completo (variante D3-A).

    Parámetros
    ----------
    mcmc_cfg : dict
        Configuración MCMC (`nsim`, `burn`, `N`, `M`).
    hp : dict
        Hiperparámetros del modelo. Ver `PSBPSampler` para la lista
        completa. Las claves `apij`, `bpij`, `mupsij`, `taupsij`
        admiten escalar o array de longitud `p` (priors heterogéneas
        por variable).
    seed : int, opcional
        Semilla de reproducibilidad.
    verbose_every : int, default=200
        Frecuencia del log de progreso. 0 = silencioso.

    Atributos post-fit
    ------------------
    sampler_ : PSBPSampler
        Instancia entrenada (acceso directo a `traces`).
    predictor_ : PSBPPredictor
        Instancia construida con las trazas.
    feature_names_ : list[str]
        Nombres de las columnas de X (sin la columna de respuesta).
    target_name_ : str
        Nombre de la columna de respuesta (la primera del DataFrame).
    n_samples_ : int
    n_features_ : int

    Contrato sobre los datos
    ------------------------
    `df_train` debe contener datos **ya estandarizados externamente**:
        - Columna 0: respuesta `y` estandarizada.
        - Columnas 1..p: covariables `X` estandarizadas (SIN columna
          de intercepto — la clase la añade internamente).

    Para `predict`, el usuario pasa X con las mismas covariables, sin
    intercepto, **estandarizadas con las MISMAS estadísticas del train**.
    Las predicciones devueltas están en la misma escala con la que
    se entrenó (estandarizada).
    """

    def __init__(self,
                 mcmc_cfg: Dict[str, int],
                 hp: Dict[str, Any],
                 seed: Optional[int] = None,
                 verbose_every: int = 200):
        self.mcmc_cfg = dict(mcmc_cfg)
        self.hp = dict(hp)
        self.seed = seed
        self.verbose_every = verbose_every

        # Atributos que se llenan en fit()
        self.sampler_: Optional[PSBPSampler] = None
        self.predictor_: Optional[PSBPPredictor] = None
        self.feature_names_: Optional[list] = None
        self.target_name_: Optional[str] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────
    # Helpers internos
    # ─────────────────────────────────────────────────────────────────────
    def _split_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Separa el DataFrame en (X, y).

        Convención: la primera columna del DataFrame es y, el resto son
        las covariables. Réplica del formato de los archivos
        `BHPin_*.txt` / `BHPout_*.txt` del proyecto.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Se esperaba pandas.DataFrame; recibido {type(df).__name__}"
            )
        if df.shape[1] < 2:
            raise ValueError(
                "El DataFrame debe tener al menos 2 columnas (y + ≥1 X)"
            )

        target_name = df.columns[0]
        feature_names = list(df.columns[1:])
        y = df.iloc[:, 0].to_numpy(dtype=np.float64)
        X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        return X, y, feature_names, target_name

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Añade columna de unos como primera columna (intercepto).

        NO estandariza. Asume que el usuario ya pasó X estandarizado
        con las estadísticas del train.
        """
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D; shape recibido {X.shape}")
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _coerce_input(self,
                      data: Union[pd.DataFrame, np.ndarray]
                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Acepta un DataFrame (con y en col 0) o un ndarray solo con X.
        Retorna (X_estandarizado_sin_intercepto, y_estandarizado_o_None).

        Importante: NO estandariza. Asume que el usuario ya pasó los
        datos estandarizados con las estadísticas del train.
        """
        if isinstance(data, pd.DataFrame):
            # Si tiene el mismo número de columnas que el train original
            # (incluyendo y), se asume formato train-like y se separa.
            if data.shape[1] == self.n_features_ + 1:
                X, y, _, _ = self._split_xy(data)
                return X, y
            # Si tiene exactamente n_features_ columnas, es solo X.
            if data.shape[1] == self.n_features_:
                return data.to_numpy(dtype=np.float64), None
            raise ValueError(
                f"DataFrame con {data.shape[1]} columnas no coincide con "
                f"n_features_={self.n_features_} (solo X) ni "
                f"{self.n_features_ + 1} (y + X)."
            )
        if isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != self.n_features_:
                raise ValueError(
                    f"ndarray debe tener shape (n, {self.n_features_}); "
                    f"recibido {data.shape}"
                )
            return data.astype(np.float64), None
        raise TypeError(
            f"Tipo no soportado: {type(data).__name__}. "
            "Use pandas.DataFrame o numpy.ndarray."
        )

    # ─────────────────────────────────────────────────────────────────────
    # API pública: fit
    # ─────────────────────────────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame) -> "PSBP_FD_v1":
        """
        Entrena el modelo PSBP-FD sobre `df_train`.

        Parámetros
        ----------
        df_train : pandas.DataFrame
            **Datos ya estandarizados externamente.** La primera columna
            es la respuesta y; las restantes son las covariables. La
            clase añade internamente la columna de intercepto al
            construir la matriz de diseño.

        Retorna
        -------
        self
        """
        # 1. Separar y, X y guardar nombres
        X, y, feature_names, target_name = self._split_xy(df_train)
        self.feature_names_ = feature_names
        self.target_name_   = target_name
        self.n_samples_     = X.shape[0]
        self.n_features_    = X.shape[1]

        # 2. Añadir intercepto (SIN estandarizar)
        X_design = self._add_intercept(X)

        # 3. Entrenar sampler
        self.sampler_ = PSBPSampler(
            mcmc_cfg      = self.mcmc_cfg,
            hp            = self.hp,
            seed          = self.seed,
            verbose_every = self.verbose_every,
        )
        self.sampler_.fit(X_design, y)

        # 4. Construir predictor (sin y_mean/y_std: opera en escala
        #    estandarizada — D3 opción A)
        self.predictor_ = PSBPPredictor(
            traces = self.sampler_.traces,
            burn   = int(self.mcmc_cfg["burn"]),
        )
        return self

    # ─────────────────────────────────────────────────────────────────────
    # API pública: predict y derivados
    # ─────────────────────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if self.predictor_ is None:
            raise RuntimeError(
                "El modelo no ha sido ajustado. Llama a fit() primero."
            )

    def predict(self,
                data: Union[pd.DataFrame, np.ndarray],
                return_std: bool = False
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predicción puntual E[y|x] **en escala estandarizada**.

        Acepta un DataFrame (con o sin columna y, ambos estandarizados)
        o un ndarray solo con X estandarizado.

        El usuario es responsable de desestandarizar la salida si
        necesita la predicción en escala original.
        """
        self._check_fitted()
        X, _ = self._coerce_input(data)
        X_design = self._add_intercept(X)
        return self.predictor_.predict(X_design, return_std=return_std)

    def predict_density(self,
                        data: Union[pd.DataFrame, np.ndarray],
                        y_grid: np.ndarray,
                        return_per_iter: bool = False
                        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Densidad condicional posterior f(y|x) sobre `y_grid`
        **en escala estandarizada**.

        `y_grid` debe estar en la misma escala estandarizada con la que
        se entrenó (es responsabilidad del usuario aplicar la misma
        transformación).
        """
        self._check_fitted()
        X, _ = self._coerce_input(data)
        X_design = self._add_intercept(X)
        return self.predictor_.predict_density(
            X_design, y_grid, return_per_iter=return_per_iter
        )

    def predict_interval(self,
                         data: Union[pd.DataFrame, np.ndarray],
                         level: float = 0.95
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Intervalo de credibilidad sobre E[y|x] **en escala estandarizada**.
        """
        self._check_fitted()
        X, _ = self._coerce_input(data)
        X_design = self._add_intercept(X)
        return self.predictor_.predict_interval(X_design, level=level)

    def inclusion_probs(self,
                        as_series: bool = False
                        ) -> Union[np.ndarray, pd.Series]:
        """
        P(γ_j = 1 | data) por variable.

        Si `as_series=True`, retorna pd.Series indexada por feature_names_.
        """
        self._check_fitted()
        incl = self.predictor_.inclusion_probs()
        if as_series:
            return pd.Series(incl, index=self.feature_names_,
                             name="inclusion_prob")
        return incl

    def rmse(self,
             data: Union[pd.DataFrame, np.ndarray],
             y_obs: Optional[np.ndarray] = None) -> float:
        """
        RMSE entre predicción puntual y `y_obs`, **en escala estandarizada**.

        Si `data` es un DataFrame con columna y estandarizada en la
        posición 0, `y_obs` se extrae automáticamente. Si es ndarray,
        debe pasarse `y_obs` explícitamente (también estandarizado).

        Si el usuario quiere RMSE en escala original, debe desestandarizar
        las predicciones y `y_obs` fuera de la clase.
        """
        self._check_fitted()
        X, y_from_df = self._coerce_input(data)
        if y_obs is None:
            if y_from_df is None:
                raise ValueError(
                    "y_obs no fue proporcionado y `data` no contiene "
                    "columna de respuesta."
                )
            y_obs = y_from_df
        X_design = self._add_intercept(X)
        return self.predictor_.rmse(X_design, y_obs)

    # ─────────────────────────────────────────────────────────────────────
    # Acceso conveniente
    # ─────────────────────────────────────────────────────────────────────
    @property
    def traces(self) -> Dict[str, np.ndarray]:
        """Diccionario de trazas MCMC (proxy a `sampler_.traces`)."""
        self._check_fitted()
        return self.sampler_.traces

    def get_config(self) -> Dict[str, Any]:
        """Configuración del modelo (útil para serialización/metadatos)."""
        # Para hp: serializar arrays como listas si los hay (priors heterogéneas)
        hp_serializable = {}
        for k, v in self.hp.items():
            if isinstance(v, np.ndarray):
                hp_serializable[k] = v.tolist()
            else:
                hp_serializable[k] = v

        return {
            "mcmc_cfg":       dict(self.mcmc_cfg),
            "hp":             hp_serializable,
            "seed":           self.seed,
            "verbose_every":  self.verbose_every,
            "feature_names":  list(self.feature_names_) if self.feature_names_ else None,
            "target_name":    self.target_name_,
            "n_samples":      self.n_samples_,
            "n_features":     self.n_features_,
            # NOTA: las estadísticas de estandarización (y_mean, y_std,
            # x_mean, x_std) ya NO se almacenan aquí porque la clase no
            # estandariza. El usuario debe gestionarlas externamente.
        }
