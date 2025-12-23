# metrics_regression.py

import numpy as np


def _validate_inputs(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompatibles: y_true {y_true.shape}, y_pred {y_pred.shape}"
        )

    if y_true.ndim != 1:
        raise ValueError("y_true y y_pred deben ser vectores 1D")

    return y_true, y_pred


def mean_squared_error(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def regression_metrics(y_true, y_pred):
    """
    Devuelve un diccionario con métricas estándar de regresión.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
