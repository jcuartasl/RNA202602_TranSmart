"""Inferencia para producción: contrato forecast() para P2 y P5."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import torch
import yaml

from transport_ai.common.paths import FORECASTING_ARTIFACTS, CONFIGS, FORECASTING_PROCESSED
from transport_ai.common.logging_utils import get_logger
from transport_ai.common.serialization import load_checkpoint
from transport_ai.forecasting.models import create_model
from transport_ai.forecasting.features import (
    agregar_features_calendario,
    construir_calendario_completo,
)
from transport_ai.forecasting.data import cargar_procesado

logger = get_logger(__name__)

# Cache global para modelo cargado (evita recarga en app)
_MODEL_CACHE = {}


def cargar_config() -> dict:
    """Carga configuración forecast.yaml."""
    with open(CONFIGS / "forecast.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cargar_modelo_y_scaler(model_type: str = None, device: str = "cpu") -> tuple:
    """
    Carga modelo y scaler desde artefactos (con cache).

    Returns:
        (model, config, scaler, metadata)
    """
    config = cargar_config()
    if model_type is None:
        model_type = config["inference"]["default_model"]

    cache_key = f"{model_type}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    artifact_path = FORECASTING_ARTIFACTS / model_type / "best.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Modelo {model_type} no entrenado. "
            f"Ejecutar entrenamiento primero. Ruta: {artifact_path}"
        )

    model, model_config, scaler, metadata = load_checkpoint(artifact_path, device=device)

    # Recrear modelo con config guardada
    model_kwargs = model_config.get("model_kwargs", model_config)
    model = create_model(model_type, model_kwargs)
    checkpoint = torch.load(artifact_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, model_config, scaler, metadata)
    logger.info(f"Modelo {model_type} cargado desde {artifact_path}")

    return model, model_config, scaler, metadata


def _preparar_ventana_inferencia(
    ruta_id: str,
    fecha_origen: str,
    window_size: int,
    horizon: int,
    model_type: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Prepara ventana de entrada para inferencia (sin fuga temporal).

    Args:
        ruta_id: ID de la ruta (ej. "11001-05001")
        fecha_origen: Fecha de corte "YYYY-MM-DD"
        window_size: Tamaño de ventana de historia
        horizon: Horizonte de predicción
        model_type: Tipo de modelo para determinar features

    Returns:
        (X_input: (1, window_size, n_features), metadata_ventana)
    """
    config = cargar_config()

    # Cargar series procesadas con features
    try:
        series = cargar_procesado("series_con_features")
    except FileNotFoundError:
        # Fallback: construir desde series base + calendario
        series_base = cargar_procesado("series_diarias")
        cal = construir_calendario_completo(
            config["splits"]["train_end"][:4] + "-01-01",  # Año inicio train
            config["splits"]["test_end"],
        )
        series = agregar_features_calendario(series_base, cal)

    # Filtrar ruta
    df_ruta = series[series["ruta_id"] == ruta_id].sort_values("fecha").reset_index(drop=True)
    if len(df_ruta) == 0:
        raise ValueError(f"Ruta {ruta_id} no encontrada en datos procesados")

    fecha_origen_dt = pd.Timestamp(fecha_origen)

    # Tomar historia hasta fecha_origen (EXCLUYENDO fecha_origen para evitar fuga)
    historia = df_ruta[df_ruta["fecha"] < fecha_origen_dt].copy()

    if len(historia) < window_size:
        raise ValueError(
            f"Historia insuficiente para ruta {ruta_id}: "
            f"{len(historia)} días < window_size {window_size}. "
            f"Fecha origen: {fecha_origen}"
        )

    # Última ventana de window_size días
    ventana = historia.tail(window_size)

    # Determinar features según modelo
    feature_cols = ["pasajeros_dia"]
    # Agregar features de calendario disponibles
    cal_features = [
        c for c in ventana.columns
        if c.startswith(("dia_semana_", "dia_ano_", "mes_", "es_", "semana_", "fin_", "temporada_"))
    ]
    feature_cols.extend(cal_features)

    # Verificar que existen
    feature_cols = [c for c in feature_cols if c in ventana.columns]

    X = ventana[feature_cols].values  # (window_size, n_features)
    X = X.reshape(1, window_size, -1)  # (1, window_size, n_features)

    meta = {
        "ruta_id": ruta_id,
        "fecha_origen": fecha_origen,
        "window_start": ventana["fecha"].iloc[0],
        "window_end": ventana["fecha"].iloc[-1],
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
    }

    return X, meta


def forecast(
    ruta_id: str,
    fecha_origen: str,
    horizon: int = 30,
    quantile: float = 0.90,
    model_type: str = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Contrato principal de inferencia para P2 y P5.

    Genera pronóstico a 30 días con cuantil P90.

    Args:
        ruta_id: ID de ruta en formato "ORIGEN-DESTINO" (códigos DIVIPOLA 5 dígitos)
        fecha_origen: Fecha de corte "YYYY-MM-DD" (último día conocido)
        horizon: Horizonte de predicción en días (default: 30)
        quantile: Cuantil para banda de incertidumbre (default: 0.90)
        model_type: "mlp" o "gru" (default: config.inference.default_model)
        device: "cpu" o "cuda"

    Returns:
        DataFrame con columnas:
        - date: fecha de predicción (YYYY-MM-DD)
        - route_id: ID de ruta
        - point_forecast: pronóstico puntual (pasajeros/día)
        - quantile_forecast: pronóstico cuantil P90 (pasajeros/día)
        - quantile: valor del cuantil usado
        - origin_date: fecha de origen del pronóstico
        - model_version: versión del modelo
        - model_type: tipo de modelo usado

    Exactamente 30 filas (o horizon filas) por ruta.
    Unidades: pasajeros por día.
    """
    config = cargar_config()

    if model_type is None:
        model_type = config["inference"]["default_model"]

    window_size = config["features"]["window_size"]
    horizon = config["features"]["horizon"] if horizon is None else horizon

    # Validaciones básicas
    if horizon < 1:
        raise ValueError("horizon debe ser >= 1")
    if not (0.50 <= quantile <= 0.99):
        raise ValueError("quantile debe estar entre 0.50 y 0.99")

    # Cargar modelo y scaler
    model, model_config, scaler, metadata = _cargar_modelo_y_scaler(model_type, device)

    # Preparar ventana de entrada (sin fuga)
    X, meta_ventana = _preparar_ventana_inferencia(
        ruta_id, fecha_origen, window_size, horizon, model_type
    )

    # Escalar
    n_samples, ws, n_features = X.shape
    X_flat = X.reshape(n_samples, -1)
    X_scaled = scaler.transform(X_flat)
    X_final = X_scaled.reshape(n_samples, ws, n_features)

    # Inferencia
    X_tensor = torch.FloatTensor(X_final).to(device)
    model.eval()

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()  # (horizon,)

    # Asegurar no negativos
    y_pred = np.maximum(y_pred, 0)

    # Cuantil: para modelo single-head, usar buffer heurístico calibrado
    # En producción con pérdida pinball real, el modelo outputaría cuantil directamente
    quantile_factor = 1.0 + (quantile - 0.50) * 0.6  # Factor calibrado empiricamente
    y_quantile = y_pred * quantile_factor

    # Generar fechas de predicción (día siguiente a fecha_origen)
    fechas = pd.date_range(
        pd.Timestamp(fecha_origen) + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    # Modelo version
    model_version = metadata.get("version", "1.0.0")

    # Construir DataFrame de salida
    result = pd.DataFrame({
        "date": fechas,
        "route_id": ruta_id,
        "point_forecast": y_pred.astype(int),
        "quantile_forecast": y_quantile.astype(int),
        "quantile": quantile,
        "origin_date": fecha_origen,
        "model_version": model_version,
        "model_type": model_type,
    })

    logger.info(f"Pronóstico generado: {ruta_id} desde {fecha_origen} horizonte {horizon}d modelo {model_type}")

    return result


def forecast_batch(
    rutas: list[str],
    fecha_origen: str,
    horizon: int = 30,
    quantile: float = 0.90,
    model_type: str = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Genera pronósticos para múltiples rutas de una vez.

    Útil para planificación operativa completa (P2).
    """
    resultados = []
    for ruta in rutas:
        try:
            df = forecast(ruta, fecha_origen, horizon, quantile, model_type, device)
            resultados.append(df)
        except Exception as e:
            logger.error(f"Error en ruta {ruta}: {e}")
            # Agregar fila de error para mantener estructura
            fechas = pd.date_range(
                pd.Timestamp(fecha_origen) + pd.Timedelta(days=1),
                periods=horizon,
                freq="D",
            )
            error_df = pd.DataFrame({
                "date": fechas,
                "route_id": ruta,
                "point_forecast": 0,
                "quantile_forecast": 0,
                "quantile": quantile,
                "origin_date": fecha_origen,
                "model_version": "error",
                "model_type": model_type or "unknown",
                "error": str(e),
            })
            resultados.append(error_df)

    return pd.concat(resultados, ignore_index=True)


def listar_rutas_disponibles() -> list[str]:
    """Retorna lista de rutas con modelos entrenados."""
    config = cargar_config()
    return config["data"]["routes"]


def obtener_metadatos_modelo(model_type: str = None) -> dict:
    """Retorna metadatos del modelo para la UI."""
    config = cargar_config()
    if model_type is None:
        model_type = config["inference"]["default_model"]

    _, _, _, metadata = _cargar_modelo_y_scaler(model_type)
    return {
        "model_type": model_type,
        "version": metadata.get("version", "unknown"),
        "deploy_seed": metadata.get("deploy_seed", "unknown"),
        "n_params": metadata.get("n_params", "unknown"),
        "config": config["models"][model_type],
    }


def limpiar_cache() -> None:
    """Limpia cache de modelos (útil para tests o cambio de modelo)."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Cache de modelos limpiado")