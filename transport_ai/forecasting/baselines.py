"""Modelos de referencia (baselines) para pronóstico de demanda."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from transport_ai.common.logging_utils import get_logger
from transport_ai.common.metrics import evaluate_forecast, evaluate_by_horizon

logger = get_logger(__name__)


def naive_seasonal_forecast(
    series: pd.DataFrame,
    ruta: str,
    fecha_origen: str,
    horizon: int = 30,
    seasonal_period: int = 7,
    quantile: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline ingenuo estacional: mismo día de la semana anterior.

    Args:
        series: DataFrame con columnas fecha, ruta_id, pasajeros_dia
        ruta: ID de la ruta
        fecha_origen: Fecha de corte "YYYY-MM-DD"
        horizon: Días a predecir
        seasonal_period: Período estacional (7=semanal)

    Returns:
        (point_forecast, quantile_forecast) arrays de longitud horizon
    """
    df_ruta = series[series["ruta_id"] == ruta].sort_values("fecha").reset_index(drop=True)
    fecha_origen = pd.Timestamp(fecha_origen)

    # Filtrar historia hasta fecha_origen (excluyendo la fecha_origen)
    historia = df_ruta[df_ruta["fecha"] < fecha_origen]

    if len(historia) < seasonal_period:
        # Fallback: usar media histórica
        media = historia["pasajeros_dia"].mean() if len(historia) > 0 else 0
        point = np.full(horizon, max(0, int(round(media))))
    else:
        # Para cada día del horizonte, usar mismo día semana anterior
        point = []
        for h in range(1, horizon + 1):
            fecha_objetivo = fecha_origen + pd.Timedelta(days=h)
            dia_semana = fecha_objetivo.dayofweek

            # Buscar último día con mismo día de semana en historia
            mismos_dia = historia[historia["fecha"].dt.dayofweek == dia_semana]
            if len(mismos_dia) > 0:
                # Usar el más reciente
                val = mismos_dia.iloc[-1]["pasajeros_dia"]
            else:
                val = historia["pasajeros_dia"].mean()

            point.append(max(0, int(round(val))))

    point = np.array(point, dtype=float)
    # Para baseline no hay modelo probabilístico: usar punto + buffer heurístico
    # Factor consistente con modelos neuronales: 1.0 + (quantile - 0.50) * 0.6
    quantile_factor = 1.0 + (quantile - 0.50) * 0.6
    quantile = point * quantile_factor

    return point, quantile


def linear_lags_forecast(
    series: pd.DataFrame,
    ruta: str,
    fecha_origen: str,
    horizon: int = 30,
    lags: list[int] = None,
    feature_cols: list[str] = None,
    quantile: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Baseline estadístico: Regresión lineal con rezagos y features de calendario.

    Args:
        series: DataFrame con features ya construidas
        ruta: ID de la ruta
        fecha_origen: Fecha de corte
        horizon: Días a predecir
        lags: Lista de rezagos a usar (ej. [1, 2, 3, 7, 14])
        feature_cols: Columnas de features adicionales

    Returns:
        (point_forecast, quantile_forecast)
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14]
    if feature_cols is None:
        feature_cols = [
            "dia_semana_sin", "dia_semana_cos",
            "dia_ano_sin", "dia_ano_cos",
            "mes_sin", "mes_cos",
            "es_festivo_o_puente", "es_temporada_alta", "temporada_escolar",
        ]

    df_ruta = series[series["ruta_id"] == ruta].sort_values("fecha").reset_index(drop=True)
    fecha_origen = pd.Timestamp(fecha_origen)

    # Filtrar historia hasta fecha_origen
    historia = df_ruta[df_ruta["fecha"] < fecha_origen].copy()

    if len(historia) < max(lags) + 10:
        logger.warning(f"Ruta {ruta}: pocos datos para linear_lags, usando media")
        media = historia["pasajeros_dia"].mean() if len(historia) > 0 else 0
        point = np.full(horizon, max(0, int(round(media))))
        return point, point * 1.2

    # Crear features de rezagos
    for lag in lags:
        historia[f"lag_{lag}"] = historia["pasajeros_dia"].shift(lag)

    # Features de calendario ya deberían estar en series
    feature_cols_exist = [c for c in feature_cols if c in historia.columns]
    all_features = [f"lag_{lag}" for lag in lags] + feature_cols_exist

    # Preparar X, y
    historia_clean = historia.dropna(subset=all_features + ["pasajeros_dia"])
    if len(historia_clean) < 20:
        media = historia["pasajeros_dia"].mean()
        point = np.full(horizon, max(0, int(round(media))))
        return point, point * 1.2

    X = historia_clean[all_features].values
    y = historia_clean["pasajeros_dia"].values

    # Escalar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar regresión lineal
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Preparar predicciones recursivas para horizonte
    # Usar últimos valores conocidos
    last_row = historia_clean.iloc[-1:].copy()
    point = []

    for h in range(horizon):
        # Actualizar rezagos en last_row (solo los que existen)
        for lag in reversed(lags):
            if lag == 1:
                last_row[f"lag_1"] = point[-1] if point else last_row["pasajeros_dia"].values[0]
            elif f"lag_{lag-1}" in last_row.columns:
                last_row[f"lag_{lag}"] = last_row[f"lag_{lag-1}"].values[0]

        # Features de calendario para fecha objetivo
        fecha_obj = fecha_origen + pd.Timedelta(days=h + 1)
        # Buscar en historia o calcular
        cal_row = historia[historia["fecha"] == fecha_obj]
        if len(cal_row) > 0:
            for c in feature_cols_exist:
                last_row[c] = cal_row[c].values[0]
        else:
            # Calcular features cíclicas básicas
            dia_sem = fecha_obj.dayofweek
            dia_ano = fecha_obj.dayofyear
            mes = fecha_obj.month
            last_row["dia_semana_sin"] = np.sin(2 * np.pi * dia_sem / 7)
            last_row["dia_semana_cos"] = np.cos(2 * np.pi * dia_sem / 7)
            last_row["dia_ano_sin"] = np.sin(2 * np.pi * dia_ano / 365.25)
            last_row["dia_ano_cos"] = np.cos(2 * np.pi * dia_ano / 365.25)
            last_row["mes_sin"] = np.sin(2 * np.pi * mes / 12)
            last_row["mes_cos"] = np.cos(2 * np.pi * mes / 12)
            # Binarias: aproximar como False
            for c in ["es_festivo_o_puente", "es_temporada_alta", "temporada_escolar"]:
                if c in last_row.columns:
                    last_row[c] = 0

        X_pred = scaler.transform(last_row[all_features].values)
        pred = max(0, model.predict(X_pred)[0])
        point.append(pred)

    point = np.array(point)
    # Factor consistente con modelos neuronales
    quantile_factor = 1.0 + (quantile - 0.50) * 0.6
    quantile = point * quantile_factor

    return point, quantile


def evaluar_baselines(
    series: pd.DataFrame,
    routes: list[str],
    fecha_origen: str,
    horizon: int = 30,
    y_train: np.ndarray = None,
) -> dict:
    """
    Evalúa ambos baselines en una fecha de origen.

    Args:
        series: Series con features
        routes: Lista de rutas
        fecha_origen: Fecha de corte
        horizon: Horizonte
        y_train: Serie de entrenamiento para MASE

    Returns:
        Diccionario con métricas por baseline y ruta
    """
    resultados = {}

    for baseline_name, baseline_fn in [
        ("naive_seasonal", naive_seasonal_forecast),
        ("linear_lags", linear_lags_forecast),
    ]:
        resultados[baseline_name] = {}
        for ruta in routes:
            point, quantile = baseline_fn(series, ruta, fecha_origen, horizon)

            # Para evaluación real, necesitaríamos y_true de test
            # Aquí solo retornamos los pronósticos
            resultados[baseline_name][ruta] = {
                "point_forecast": point.tolist(),
                "quantile_forecast": quantile.tolist(),
            }

    return resultados


def forecast_baseline(
    series: pd.DataFrame,
    ruta: str,
    fecha_origen: str,
    horizon: int = 30,
    method: str = "naive_seasonal",
    quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Genera pronóstico con baseline especificado en formato estándar.

    Returns:
        DataFrame con columnas: date, route_id, point_forecast, quantile_forecast, quantile, model
    """
    if method == "naive_seasonal":
        point, quant = naive_seasonal_forecast(series, ruta, fecha_origen, horizon)
    elif method == "linear_lags":
        point, quant = linear_lags_forecast(series, ruta, fecha_origen, horizon)
    else:
        raise ValueError(f"Baseline desconocido: {method}")

    fechas = pd.date_range(
        pd.Timestamp(fecha_origen) + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    return pd.DataFrame({
        "date": fechas,
        "route_id": ruta,
        "point_forecast": point,
        "quantile_forecast": quant,
        "quantile": quantile,
        "model": f"baseline_{method}",
    })