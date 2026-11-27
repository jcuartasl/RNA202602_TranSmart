"""Evaluación completa de modelos: métricas, comparativas, reportes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
import torch
import yaml

from transport_ai.common.paths import FORECASTING_ARTIFACTS, EXPERIMENTS, CONFIGS
from transport_ai.common.logging_utils import get_logger
from transport_ai.common.metrics import (
    evaluate_forecast,
    evaluate_by_horizon,
    mae,
    rmse,
    mase,
    pinball_loss,
    coverage,
)
from transport_ai.common.serialization import load_checkpoint, load_json
from transport_ai.forecasting.models import create_model
from transport_ai.forecasting.baselines import forecast_baseline

logger = get_logger(__name__)


def cargar_config() -> dict:
    """Carga configuración forecast.yaml."""
    with open(CONFIGS / "forecast.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_modelo_entrenado(model_type: str, device: str = "cpu") -> tuple:
    """
    Carga modelo entrenado (best.pt) con su scaler y config.

    Returns:
        (model, config, scaler, metadata)
    """
    artifact_path = FORECASTING_ARTIFACTS / model_type / "best.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {artifact_path}")

    model, config, scaler, metadata = load_checkpoint(artifact_path, device=device)

    # Recrear modelo con config guardada
    model_kwargs = config.get("model_kwargs", config)
    model = create_model(model_type, model_kwargs)
    model.load_state_dict(torch.load(artifact_path, map_location=device, weights_only=False)["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config, scaler, metadata


def predecir_modelo(
    model: torch.nn.Module,
    X: np.ndarray,
    scaler: Any,
    device: str = "cpu",
) -> np.ndarray:
    """
    Genera predicciones con modelo entrenado.

    Args:
        model: Modelo en modo eval
        X: (n_samples, window_size, n_features)
        scaler: Scaler ajustado en entrenamiento
        device: Dispositivo

    Returns:
        y_pred: (n_samples, horizon)
    """
    n_samples, window_size, n_features = X.shape
    X_flat = X.reshape(n_samples, -1)
    X_scaled = scaler.transform(X_flat)
    X_final = X_scaled.reshape(n_samples, window_size, n_features)

    X_tensor = torch.FloatTensor(X_final).to(device)

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    return y_pred


def evaluar_modelo_completo(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    metadata: pd.DataFrame,
    y_train_series: np.ndarray = None,
    device: str = "cpu",
) -> dict:
    """
    Evaluación completa de un modelo entrenado en test.

    Args:
        model_type: "mlp" o "gru"
        X, y, splits, metadata: Datos de test
        y_train_series: Serie de entrenamiento para MASE (1D array)

    Returns:
        Diccionario con métricas completas
    """
    # Cargar modelo
    model, config, scaler, _ = cargar_modelo_entrenado(model_type, device)

    # Predecir en test
    test_indices = splits["test"]
    y_pred = predecir_modelo(model, X[test_indices], scaler, device)
    y_true = y[test_indices]

    # Métricas agregadas
    config_eval = cargar_config()["evaluation"]
    quantile = config_eval.get("target_coverage", 0.90)
    seasonal_period = config_eval.get("mase_seasonal_period", 7)

    # Para P90: usar point * factor (ya que modelo single-head no tiene cuantil separado)
    # En producción real, modelo tendría cabeza cuantil o loss pinball
    # Factor consistente con infer.py: 1.0 + (quantile - 0.50) * 0.6
    quantile_factor = 1.0 + (quantile - 0.50) * 0.6
    y_pred_quantile = y_pred * quantile_factor

    # MASE necesita serie de entrenamiento completa por ruta (no media por ventana)
    # Obtener series de entrenamiento reales desde metadata
    if y_train_series is None:
        # Usar las series originales de entrenamiento para MASE
        # Aquí aproximamos con la media por ventana, pero lo ideal sería la serie completa
        y_train_series = y[splits["train"]].mean(axis=1)

    metrics = evaluate_forecast(
        y_true, y_pred, y_pred_quantile, y_train_series,
        quantile=quantile, seasonal_period=seasonal_period
    )

    # Métricas por tramos de horizonte
    segments = config_eval.get("horizon_segments", [[1, 7], [8, 14], [15, 30]])
    by_horizon = evaluate_by_horizon(
        y_true, y_pred, y_pred_quantile, y_train_series,
        segments=segments, quantile=quantile, seasonal_period=seasonal_period
    )

    # Métricas por ruta
    rutas_test = metadata.iloc[test_indices]["ruta_id"].values
    metricas_por_ruta = {}
    for ruta in np.unique(rutas_test):
        mask = rutas_test == ruta
        if mask.sum() > 0:
            y_train_ruta = y[splits["train"]][
                np.array([r == ruta for r in metadata.iloc[splits["train"]]["ruta_id"].values])
            ].mean(axis=1) if len(splits["train"]) > 0 else y_train_series

            metricas_por_ruta[ruta] = evaluate_forecast(
                y_true[mask], y_pred[mask], y_pred_quantile[mask], y_train_ruta,
                quantile=quantile, seasonal_period=seasonal_period
            )

    return {
        "model_type": model_type,
        "metrics": metrics,
        "by_horizon": by_horizon,
        "by_route": metricas_por_ruta,
        "n_samples": len(y_true),
        "n_params": model.get_num_params(),
    }


def comparar_modelos(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    metadata: pd.DataFrame,
    model_types: list[str] = None,
    routes: list[str] = None,
    y_train_series: np.ndarray = None,
    device: str = "cpu",
) -> dict:
    """
    Compara múltiples modelos (neuronales + baselines) en mismas particiones.

    Returns:
        Diccionario con resultados de todos los modelos
    """
    if model_types is None:
        model_types = ["mlp", "gru"]

    config = cargar_config()
    eval_config = config["evaluation"]
    quantile = eval_config.get("target_coverage", 0.90)
    seasonal_period = eval_config.get("mase_seasonal_period", 7)
    segments = eval_config.get("horizon_segments", [[1, 7], [8, 14], [15, 30]])

    # Serie de entrenamiento para MASE
    if y_train_series is None:
        y_train_series = y[splits["train"]].mean(axis=1)

    resultados = {}

    # 1. Modelos neuronales
    for model_type in model_types:
        try:
            logger.info(f"Evaluando {model_type}...")
            resultados[model_type] = evaluar_modelo_completo(
                model_type, X, y, splits, metadata,
                y_train_series=y_train_series, device=device
            )
        except FileNotFoundError:
            logger.warning(f"Modelo {model_type} no encontrado, saltando")
            resultados[model_type] = {"error": "modelo_no_encontrado"}

    # 2. Baselines (en fechas de origen de test)
    # Obtener fechas de origen únicas en test
    test_meta = metadata.iloc[splits["test"]]
    fechas_origen_test = test_meta["fecha_origen"].unique()

    # Usar primera fecha de origen representativa para baseline
    # (En práctica se evaluaría en cada origen rodante)
    if len(fechas_origen_test) > 0:
        fecha_rep = fechas_origen_test[0].strftime("%Y-%m-%d")
        logger.info(f"Evaluando baselines en origen {fecha_rep}...")

        # Cargar series para baselines
        from transport_ai.forecasting.features import agregar_features_calendario
        from transport_ai.forecasting.data import cargar_procesado

        try:
            series = cargar_procesado("series_con_features")
            if routes is None:
                routes = series["ruta_id"].unique().tolist()

            for baseline_name in ["naive_seasonal", "linear_lags"]:
                logger.info(f"  Baseline: {baseline_name}")
                baseline_results = {"by_route": {}}

                for ruta in routes:
                    # Generar pronóstico baseline
                    fc_df = forecast_baseline(
                        series, ruta, fecha_rep,
                        horizon=config["features"]["horizon"],
                        method=baseline_name,
                        quantile=quantile,
                    )

                    # Para evaluar, necesitamos y_true de test para esa ruta/origen
                    # Simplificación: usar primeras muestras de test de esa ruta
                    ruta_mask = (metadata.iloc[splits["test"]]["ruta_id"] == ruta).values
                    if ruta_mask.sum() > 0:
                        idx_ruta = np.where(ruta_mask)[0]
                        y_true_ruta = y[splits["test"]][idx_ruta]
                        y_pred_ruta = fc_df["point_forecast"].values[:len(idx_ruta)]
                        y_quant_ruta = fc_df["quantile_forecast"].values[:len(idx_ruta)]

                        y_train_ruta = y[splits["train"]][
                            np.array([r == ruta for r in metadata.iloc[splits["train"]]["ruta_id"].values])
                        ].mean(axis=1) if len(splits["train"]) > 0 else y_train_series

                        baseline_results["by_route"][ruta] = evaluate_forecast(
                            y_true_ruta, y_pred_ruta, y_quant_ruta, y_train_ruta,
                            quantile=quantile, seasonal_period=seasonal_period
                        )

                # Agregados
                all_true = []
                all_pred = []
                all_quant = []
                for ruta in routes:
                    ruta_mask = (metadata.iloc[splits["test"]]["ruta_id"] == ruta).values
                    if ruta_mask.sum() > 0:
                        idx_ruta = np.where(ruta_mask)[0]
                        fc_df = forecast_baseline(
                            series, ruta, fecha_rep,
                            horizon=config["features"]["horizon"],
                            method=baseline_name,
                            quantile=quantile,
                        )
                        all_true.append(y[splits["test"]][idx_ruta])
                        all_pred.append(fc_df["point_forecast"].values[:len(idx_ruta)])
                        all_quant.append(fc_df["quantile_forecast"].values[:len(idx_ruta)])

                if all_true:
                    y_true_all = np.vstack(all_true)
                    y_pred_all = np.vstack(all_pred)
                    y_quant_all = np.vstack(all_quant)

                    baseline_results["metrics"] = evaluate_forecast(
                        y_true_all, y_pred_all, y_quant_all, y_train_series,
                        quantile=quantile, seasonal_period=seasonal_period
                    )
                    baseline_results["by_horizon"] = evaluate_by_horizon(
                        y_true_all, y_pred_all, y_quant_all, y_train_series,
                        segments=segments, quantile=quantile, seasonal_period=seasonal_period
                    )

                resultados[f"baseline_{baseline_name}"] = baseline_results

        except Exception as e:
            logger.warning(f"Error evaluando baselines: {e}")

    return resultados


def generar_reporte_comparativo(
    resultados: dict,
    output_path: Path,
) -> None:
    """Genera reporte JSON comparativo de todos los modelos."""
    # Extraer métricas principales para tabla resumen
    resumen = {}
    for model_name, result in resultados.items():
        if "error" in result:
            resumen[model_name] = {"error": result["error"]}
            continue

        metrics = result.get("metrics", {})
        resumen[model_name] = {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mase": metrics.get("mase"),
            "pinball": metrics.get("pinball"),
            "coverage": metrics.get("coverage"),
            "n_params": result.get("n_params", "N/A"),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "resumen": resumen,
            "detalle": resultados,
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Reporte comparativo guardado en {output_path}")


def cargar_resultados_30_semillas(model_type: str) -> dict:
    """Carga resultados de las 30 semillas."""
    path = FORECASTING_ARTIFACTS / model_type / "seeds_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Resultados de semillas no encontrados: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resumen_30_semillas(model_type: str) -> dict:
    """Genera resumen estadístico de 30 semillas."""
    results = cargar_resultados_30_semillas(model_type)

    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    val_losses = [r["val_loss_final"] for r in results]
    times = [r["train_time_sec"] for r in results]

    return {
        "model_type": model_type,
        "n_seeds": len(results),
        "mae": {
            "mean": float(np.mean(maes)),
            "std": float(np.std(maes, ddof=1)),
            "min": float(np.min(maes)),
            "max": float(np.max(maes)),
        },
        "rmse": {
            "mean": float(np.mean(rmses)),
            "std": float(np.std(rmses, ddof=1)),
            "min": float(np.min(rmses)),
            "max": float(np.max(rmses)),
        },
        "val_loss": {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses, ddof=1)),
        },
        "train_time_sec": {
            "mean": float(np.mean(times)),
            "total": float(np.sum(times)),
        },
    }