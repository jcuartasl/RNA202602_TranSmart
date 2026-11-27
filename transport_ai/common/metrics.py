"""Métricas de evaluación para series temporales y pronósticos."""

import numpy as np
import torch
from typing import Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Error absoluto medio."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 7,
) -> float:
    """
    MASE (Mean Absolute Scaled Error) con denominador estacional.

    MASE = MAE_test / MAE_naive_seasonal_train
    Donde MAE_naive_seasonal_train = mean(|y_t - y_{t-m}|) para t=m+1..T en train

    Args:
        y_true: Valores reales de test
        y_pred: Valores predichos de test
        y_train: Valores de entrenamiento (para calcular denominador)
        seasonal_period: Período estacional (7=día semanal, 365=anual)

    Returns:
        MASE score (menor es mejor, <1 supera baseline estacional)
    """
    # MAE del test
    mae_test = np.mean(np.abs(y_true - y_pred))

    # Denominador: MAE del baseline estacional en train
    if len(y_train) <= seasonal_period:
        # Serie muy corta: usar MAE simple como fallback
        denom = np.mean(np.abs(np.diff(y_train)))
        if denom == 0:
            return float("inf")
        return float(mae_test / denom)

    # Error del baseline estacional (mismo día semana anterior)
    seasonal_errors = np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])
    denom = np.mean(seasonal_errors)

    if denom == 0:
        # Serie constante en train
        return float("inf") if mae_test > 0 else 0.0

    return float(mae_test / denom)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.90) -> float:
    """
    Pérdida pinball (quantile loss) para evaluación de cuantil.

    L_q(y, q) = (y - q) * (quantile - I(y < q))
    Donde I es la función indicadora.

    Args:
        y_true: Valores reales
        y_pred: Cuantil predicho (ej. P90)
        quantile: Cuantil objetivo (0.90 para P90)

    Returns:
        Pinball loss promedio
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return float(np.mean(loss))


def coverage(y_true: np.ndarray, y_pred_upper: np.ndarray) -> float:
    """
    Cobertura empírica del intervalo de predicción.

    Fracción de observaciones donde y_true <= y_pred_upper.
    Para P90 bien calibrado, debería ser ~0.90.
    """
    return float(np.mean(y_true <= y_pred_upper))


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred_point: np.ndarray,
    y_pred_quantile: np.ndarray,
    y_train: np.ndarray,
    quantile: float = 0.90,
    seasonal_period: int = 7,
) -> dict:
    """
    Evaluación completa de un pronóstico puntual + cuantil.

    Returns:
        Diccionario con todas las métricas
    """
    return {
        "mae": mae(y_true, y_pred_point),
        "rmse": rmse(y_true, y_pred_point),
        "mase": mase(y_true, y_pred_point, y_train, seasonal_period),
        "pinball": pinball_loss(y_true, y_pred_quantile, quantile),
        "coverage": coverage(y_true, y_pred_quantile),
    }


def evaluate_by_horizon(
    y_true: np.ndarray,
    y_pred_point: np.ndarray,
    y_pred_quantile: np.ndarray,
    y_train: np.ndarray,
    segments: list[tuple[int, int]],
    quantile: float = 0.90,
    seasonal_period: int = 7,
) -> dict:
    """
    Evalúa métricas por tramos de horizonte (ej. días 1-7, 8-14, 15-30).

    Args:
        segments: Lista de tuplas (inicio, fin) 1-indexed, inclusive
    """
    results = {}
    for start, end in segments:
        # Convertir a 0-indexed
        s, e = start - 1, end
        key = f"h{start}_{end}"
        results[key] = evaluate_forecast(
            y_true[s:e],
            y_pred_point[s:e],
            y_pred_quantile[s:e],
            y_train,
            quantile,
            seasonal_period,
        )
    return results