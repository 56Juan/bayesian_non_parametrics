"""
PSBP_FD_v1 — Clase orquestadora del modelo Probit Stick-Breaking Process
=========================================================================

Compone los componentes funcionales del paquete:

    pd.DataFrame ──► estandarización interna ──► PSBPSampler ──► PSBPPredictor

Esta clase ofrece una interfaz estilo sklearn (fit/predict) y oculta los
detalles de:
    - Separar y de X.
    - Estandarizar usando estadísticas del train.
    - Añadir la columna de intercepto.
    - Reusar las mismas estadísticas al hacer predict sobre datos nuevos.

NOTA: la estandarización vive aquí como código directo. En una versión
posterior se moverá a `model_psbp_fd/functions_models/functions_standarize.py`.

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
    Modelo PSBP-FD versión 1 — orquestador completo.

    Parámetros
    ----------
    mcmc_cfg : dict
        Configuración MCMC (`nsim`, `burn`, `N`, `M`).
    hp : dict
        Hiperparámetros del modelo. Ver `PSBPSampler` para la lista completa.
    seed : int, opcional
        Semilla de reproducibilidad.
    verbose_every : int, default=200
        Frecuencia del log de progreso. 0 = silencioso.

    Atributos post-fit
    ------------------
    sampler_ : PSBPSampler
        Instancia entrenada (acceso directo a `traces`).
    predictor_ : PSBPPredictor
        Instancia construida con las trazas + parámetros de desestandarización.
    feature_names_ : list[str]
        Nombres de las columnas de X (sin la columna de respuesta).
    target_name_ : str
        Nombre de la columna de respuesta (la primera del DataFrame).
    x_mean_, x_std_ : ndarray, shape (p,)
        Estadísticas de estandarización del train para X.
    y_mean_, y_std_ : float
        Estadísticas de estandarización del train para y.
    n_samples_ : int
    n_features_ : int
    """

    def __init__(self,
                 mcmc_cfg: Dict[str, int],
                 hp: Dict[str, float],
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
        self.x_mean_: Optional[np.ndarray] = None
        self.x_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None

    # ─────────────────────────────────────────────────────────────────────
    # Helpers internos
    # ─────────────────────────────────────────────────────────────────────
    def _split_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list, str]:
        """
        Separa el DataFrame en (X, y).

        Convención: la primera columna del DataFrame es y, el resto son las
        covariables. Replica el formato de los archivos `BHPin_*.txt` /
        `BHPout_*.txt` del proyecto.
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

    def _standardize_train(self,
                           X: np.ndarray,
                           y: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula y guarda estadísticas del train; retorna (X_std, y_std).

        Usa ddof=0 (consistente con `std(X, 1)` de MATLAB y con el notebook
        de referencia).
        """
        self.x_mean_ = X.mean(axis=0)
        self.x_std_  = X.std(axis=0, ddof=0)
        self.y_mean_ = float(y.mean())
        self.y_std_  = float(y.std(ddof=0))

        # Salvaguarda contra columnas constantes
        if np.any(self.x_std_ == 0):
            zero_cols = np.where(self.x_std_ == 0)[0]
            raise ValueError(
                f"Columnas con varianza cero en X: índices {zero_cols.tolist()}"
            )
        if self.y_std_ == 0:
            raise ValueError("La respuesta y tiene varianza cero")

        X_std = (X - self.x_mean_) / self.x_std_
        y_std = (y - self.y_mean_) / self.y_std_
        return X_std, y_std

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        """
        Estandariza X con las estadísticas del train y añade intercepto.
        """
        if self.x_mean_ is None:
            raise RuntimeError(
                "El modelo no ha sido ajustado. Llama a fit() primero."
            )
        X_std = (X - self.x_mean_) / self.x_std_
        return np.hstack([np.ones((X_std.shape[0], 1)), X_std])

    def _coerce_input(self,
                      data: Union[pd.DataFrame, np.ndarray]
                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Acepta un DataFrame (con y en col 0) o un ndarray solo con X.
        Retorna (X_raw, y_raw_or_None).
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
            La primera columna es la respuesta y; las restantes son las
            covariables. Réplica del formato `BHPin_*.txt`.

        Retorna
        -------
        self
        """
        # 1. Separar y, X y guardar nombres
        X_raw, y_raw, feature_names, target_name = self._split_xy(df_train)
        self.feature_names_ = feature_names
        self.target_name_   = target_name
        self.n_samples_     = X_raw.shape[0]
        self.n_features_    = X_raw.shape[1]

        # 2. Estandarizar (guarda estadísticas en self.*_mean_ / *_std_)
        X_std, y_std = self._standardize_train(X_raw, y_raw)

        # 3. Añadir intercepto
        X_design = np.hstack([np.ones((X_std.shape[0], 1)), X_std])

        # 4. Entrenar sampler
        self.sampler_ = PSBPSampler(
            mcmc_cfg      = self.mcmc_cfg,
            hp            = self.hp,
            seed          = self.seed,
            verbose_every = self.verbose_every,
        )
        self.sampler_.fit(X_design, y_std)

        # 5. Construir predictor
        self.predictor_ = PSBPPredictor(
            traces = self.sampler_.traces,
            burn   = int(self.mcmc_cfg["burn"]),
            y_mean = self.y_mean_,
            y_std  = self.y_std_,
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
        Predicción puntual E[y|x] en escala original.

        Acepta un DataFrame (con o sin columna y) o un ndarray solo con X.
        """
        self._check_fitted()
        X_raw, _ = self._coerce_input(data)
        X_design = self._transform_X(X_raw)
        return self.predictor_.predict(X_design, return_std=return_std)

    def predict_density(self,
                        data: Union[pd.DataFrame, np.ndarray],
                        y_grid: np.ndarray,
                        return_per_iter: bool = False
                        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Densidad condicional posterior f(y|x) sobre `y_grid` (escala original).
        """
        self._check_fitted()
        X_raw, _ = self._coerce_input(data)
        X_design = self._transform_X(X_raw)
        return self.predictor_.predict_density(
            X_design, y_grid, return_per_iter=return_per_iter
        )

    def predict_interval(self,
                         data: Union[pd.DataFrame, np.ndarray],
                         level: float = 0.95
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Intervalo de credibilidad sobre E[y|x].
        """
        self._check_fitted()
        X_raw, _ = self._coerce_input(data)
        X_design = self._transform_X(X_raw)
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
        RMSE entre predicción puntual y `y_obs`, en escala original.

        Si `data` es un DataFrame con columna y en la posición 0, `y_obs`
        se extrae automáticamente. Si es ndarray, debe pasarse `y_obs`
        explícitamente.
        """
        self._check_fitted()
        X_raw, y_from_df = self._coerce_input(data)
        if y_obs is None:
            if y_from_df is None:
                raise ValueError(
                    "y_obs no fue proporcionado y `data` no contiene "
                    "columna de respuesta."
                )
            y_obs = y_from_df
        X_design = self._transform_X(X_raw)
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
        return {
            "mcmc_cfg":       dict(self.mcmc_cfg),
            "hp":             dict(self.hp),
            "seed":           self.seed,
            "verbose_every":  self.verbose_every,
            "feature_names":  list(self.feature_names_) if self.feature_names_ else None,
            "target_name":    self.target_name_,
            "n_samples":      self.n_samples_,
            "n_features":     self.n_features_,
            "x_mean":         self.x_mean_.tolist() if self.x_mean_ is not None else None,
            "x_std":          self.x_std_.tolist()  if self.x_std_  is not None else None,
            "y_mean":         self.y_mean_,
            "y_std":          self.y_std_,
        }
