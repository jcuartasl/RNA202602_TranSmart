"""CLI para P1 - Pronóstico de demanda y modelos temporales."""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

from transport_ai.common.paths import ensure_dirs
from transport_ai.common.logging_utils import setup_logging, get_logger
from transport_ai.forecasting.data import (
    cargar_dataset_crudo,
    normalizar_columnas,
    guardar_procesado,
    generar_manifiesto,
)
from transport_ai.forecasting.features import (
    construir_series_diarias,
    detectar_anomalias,
    agregar_features_calendario,
    construir_calendario_completo,
    crear_ventanas_deslizantes,
    cargar_config,
)
from transport_ai.forecasting.splits import (
    crear_particiones_fijas,
    crear_origenes_rodantes,
    guardar_splits,
    verificar_sin_fuga,
)
from transport_ai.forecasting.baselines import forecast_baseline, evaluar_baselines
from transport_ai.forecasting.train import (
    buscar_hiperparametros,
    entrenar_30_semillas,
)
from transport_ai.forecasting.evaluation import (
    comparar_modelos,
    generar_reporte_comparativo,
    resumen_30_semillas,
)
from transport_ai.forecasting.infer import forecast, forecast_batch, listar_rutas_disponibles

logger = get_logger(__name__)


def cmd_preparar_datos(args: argparse.Namespace) -> int:
    """Paso 1: Cargar dataset crudo, validar, normalizar, guardar."""
    setup_logging()
    logger.info("=== PREPARAR DATOS ===")

    config = cargar_config()
    raw_path = Path(config["data"]["raw_path"])

    # Cargar y validar
    df = cargar_dataset_crudo(raw_path)

    # Guardar normalizado
    guardar_procesado(df, "dataset_normalizado")

    # Generar manifiesto
    routes = config["data"]["routes"]
    manifest_path = Path(config["data"]["manifest_path"])
    generar_manifiesto(raw_path, Path(config["data"]["processed_dir"]), routes, manifest_path)

    logger.info("Datos preparados correctamente")
    return 0


def cmd_construir_series(args: argparse.Namespace) -> int:
    """Paso 2: Construir series diarias por ruta."""
    setup_logging()
    logger.info("=== CONSTRUIR SERIES DIARIAS ===")

    config = cargar_config()
    from transport_ai.forecasting.data import cargar_procesado

    df = cargar_procesado("dataset_normalizado")
    routes = config["data"]["routes"]

    # Construir series
    missing_policy = config["features"]["missing_days_policy"]
    series = construir_series_diarias(df, routes, missing_policy)

    # Detectar anomalías
    series = detectar_anomalias(
        series,
        config["features"]["anomaly_detection"]["max_passengers_per_dispatch"],
        config["features"]["anomaly_detection"]["zero_dispatch_threshold"],
    )

    # Guardar
    guardar_procesado(series, "series_diarias")

    logger.info("Series diarias construidas")
    return 0


def cmd_construir_features(args: argparse.Namespace) -> int:
    """Paso 3: Agregar features de calendario y crear ventanas."""
    setup_logging()
    logger.info("=== CONSTRUIR FEATURES ===")

    config = cargar_config()
    from transport_ai.forecasting.data import cargar_procesado

    series = cargar_procesado("series_diarias")

    # Calendario
    splits_cfg = config["splits"]
    cal = construir_calendario_completo(splits_cfg["train_end"][:4] + "-01-01", splits_cfg["test_end"])
    guardar_procesado(cal, "calendario")

    # Features de calendario en series
    series_con_features = agregar_features_calendario(series, cal)
    guardar_procesado(series_con_features, "series_con_features")

    # Ventanas deslizantes
    window_size = config["features"]["window_size"]
    horizon = config["features"]["horizon"]
    X, y, rutas, metadata = crear_ventanas_deslizantes(
        series_con_features, window_size, horizon
    )

    # Guardar arrays
    from transport_ai.common.serialization import save_numpy
    from transport_ai.common.paths import FORECASTING_PROCESSED

    save_numpy(X, FORECASTING_PROCESSED / "X.npy")
    save_numpy(y, FORECASTING_PROCESSED / "y.npy")
    save_numpy(rutas, FORECASTING_PROCESSED / "rutas.npy")
    metadata.to_parquet(FORECASTING_PROCESSED / "metadata.parquet", index=False)

    logger.info(f"Ventanas creadas: X={X.shape}, y={y.shape}")
    return 0


def cmd_crear_splits(args: argparse.Namespace) -> int:
    """Paso 4: Crear particiones temporales fijas y rolling origin."""
    setup_logging()
    logger.info("=== CREAR PARTICIONES ===")

    config = cargar_config()
    from transport_ai.common.paths import FORECASTING_PROCESSED
    from transport_ai.common.serialization import load_numpy

    metadata = pd.read_parquet(FORECASTING_PROCESSED / "metadata.parquet")

    splits_cfg = config["splits"]
    splits_fijos = crear_particiones_fijas(
        metadata,
        splits_cfg["train_end"],
        splits_cfg["val_start"],
        splits_cfg["val_end"],
        splits_cfg["test_start"],
        splits_cfg["test_end"],
    )

    splits_rodantes = crear_origenes_rodantes(
        metadata,
        splits_cfg["rolling_origins"],
        config["features"]["horizon"],
    )

    # Verificar sin fuga
    ok = verificar_sin_fuga(splits_fijos, splits_rodantes, metadata)
    if not ok:
        logger.error("VERIFICACIÓN ANTI-FUGA FALLÓ")
        return 1

    # Guardar
    output_path = Path(config["data"]["processed_dir"]) / "splits.json"
    guardar_splits(splits_fijos, splits_rodantes, output_path)

    logger.info("Particiones creadas y verificadas")
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Evalúa baselines en particiones."""
    setup_logging()
    logger.info("=== EVALUAR BASELINES ===")

    config = cargar_config()
    from transport_ai.common.paths import FORECASTING_PROCESSED
    from transport_ai.common.serialization import load_numpy

    X = load_numpy(FORECASTING_PROCESSED / "X.npy")
    y = load_numpy(FORECASTING_PROCESSED / "y.npy")
    metadata = pd.read_parquet(FORECASTING_PROCESSED / "metadata.parquet")

    splits_path = Path(config["data"]["processed_dir"]) / "splits.json"
    import json
    with open(splits_path) as f:
        splits_data = json.load(f)

    splits = {k: np.array(v) for k, v in splits_data["splits_fijos"].items()}

    # Evaluar en test (usar primera fecha origen representativa)
    # En práctica se evaluaría en cada rolling origin
    fecha_test = config["splits"]["test_start"]
    routes = config["data"]["routes"]

    from transport_ai.forecasting.features import agregar_features_calendario
    series = cargar_procesado("series_con_features")

    resultados = evaluar_baselines(series, routes, fecha_test, config["features"]["horizon"])

    # Guardar
    output = Path(config["experiments_dir"]) / "baselines_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(resultados, f, indent=2, default=str)

    logger.info(f"Resultados baselines guardados en {output}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    """Búsqueda de hiperparámetros."""
    setup_logging()
    logger.info("=== BÚSQUEDA HIPERPARÁMETROS ===")

    config = cargar_config()
    model_type = args.model or "mlp"

    from transport_ai.common.paths import FORECASTING_PROCESSED
    from transport_ai.common.serialization import load_numpy

    X = load_numpy(FORECASTING_PROCESSED / "X.npy")
    y = load_numpy(FORECASTING_PROCESSED / "y.npy")
    metadata = pd.read_parquet(FORECASTING_PROCESSED / "metadata.parquet")

    splits_path = Path(config["data"]["processed_dir"]) / "splits.json"
    import json
    with open(splits_path) as f:
        splits_data = json.load(f)

    splits = {k: np.array(v) for k, v in splits_data["splits_fijos"].items()}

    search_space = config["models"]["hyperparameter_search"][model_type]
    n_trials = args.trials or config["models"]["hyperparameter_search"]["n_trials"]

    best_config = buscar_hiperparametros(X, y, splits, model_type, search_space, n_trials)

    logger.info(f"Mejor config: {best_config}")
    return 0


def cmd_entrenar(args: argparse.Namespace) -> int:
    """Entrena modelo final con 30 semillas."""
    setup_logging()
    logger.info("=== ENTRENAR 30 SEMILLAS ===")

    config = cargar_config()
    model_type = args.model or "mlp"

    from transport_ai.common.paths import FORECASTING_PROCESSED
    from transport_ai.common.serialization import load_numpy

    X = load_numpy(FORECASTING_PROCESSED / "X.npy")
    y = load_numpy(FORECASTING_PROCESSED / "y.npy")
    metadata = pd.read_parquet(FORECASTING_PROCESSED / "metadata.parquet")

    splits_path = Path(config["data"]["processed_dir"]) / "splits.json"
    import json
    with open(splits_path) as f:
        splits_data = json.load(f)

    splits = {k: np.array(v) for k, v in splits_data["splits_fijos"].items()}

    model_config = config["models"][model_type]
    seeds = list(range(config["models"]["seeds"]["start_seed"],
                       config["models"]["seeds"]["start_seed"] + config["models"]["seeds"]["count"]))
    deploy_seed = config["models"]["seeds"]["deploy_seed"]

    resultado = entrenar_30_semillas(
        X, y, splits, model_type, model_config,
        seeds=seeds, deploy_seed=deploy_seed
    )

    logger.info(f"Entrenamiento completado: {resultado['summary']}")
    return 0


def cmd_evaluar(args: argparse.Namespace) -> int:
    """Evalúa modelos entrenados en test."""
    setup_logging()
    logger.info("=== EVALUAR MODELOS ===")

    config = cargar_config()

    from transport_ai.common.paths import FORECASTING_PROCESSED
    from transport_ai.common.serialization import load_numpy

    X = load_numpy(FORECASTING_PROCESSED / "X.npy")
    y = load_numpy(FORECASTING_PROCESSED / "y.npy")
    metadata = pd.read_parquet(FORECASTING_PROCESSED / "metadata.parquet")

    splits_path = Path(config["data"]["processed_dir"]) / "splits.json"
    import json
    with open(splits_path) as f:
        splits_data = json.load(f)

    splits = {k: np.array(v) for k, v in splits_data["splits_fijos"].items()}

    resultados = comparar_modelos(X, y, splits, metadata)

    output = Path(config["experiments_dir"]) / "comparacion_modelos.json"
    generar_reporte_comparativo(resultados, output)

    # Resumen 30 semillas
    for model_type in ["mlp", "gru"]:
        try:
            resumen = resumen_30_semillas(model_type)
            logger.info(f"Resumen {model_type}: MAE={resumen['mae']['mean']:.2f}±{resumen['mae']['std']:.2f}")
        except FileNotFoundError:
            pass

    return 0


def cmd_predecir(args: argparse.Namespace) -> int:
    """Genera pronóstico para una ruta (inferencia)."""
    setup_logging()
    logger.info("=== PRONÓSTICO (INFERENCIA) ===")

    resultado = forecast(
        args.ruta,
        args.fecha_origen,
        horizon=args.horizon,
        quantile=args.quantile,
        model_type=args.model,
    )

    print(resultado.to_json(orient="records", date_format="iso"))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Modo demo rápido para sustentación (< 120s)."""
    setup_logging()
    logger.info("=== MODO DEMO ===")

    config = cargar_config()
    routes = config["data"]["routes"]
    fecha_demo = config["splits"]["test_start"]  # Primera fecha de test

    logger.info(f"Demo: pronóstico para {len(routes)} rutas desde {fecha_demo}")

    resultado = forecast_batch(routes, fecha_demo, model_type="gru")

    print(f"Pronósticos generados: {len(resultado)} filas")
    print(f"Rutas: {resultado['route_id'].unique()}")
    print(f"Rango fechas: {resultado['date'].min()} a {resultado['date'].max()}")

    # Métricas rápidas
    print(f"Total pasajeros (puntual 30d): {resultado['point_forecast'].sum():,}")
    print(f"Total pasajeros (P90 30d): {resultado['quantile_forecast'].sum():,}")

    return 0


def cmd_full(args: argparse.Namespace) -> int:
    """Ejecuta pipeline completo: datos -> features -> splits -> baselines -> tune -> train -> eval."""
    logger.info("=== PIPELINE COMPLETO ===")

    steps = [
        ("preparar_datos", cmd_preparar_datos),
        ("construir_series", cmd_construir_series),
        ("construir_features", cmd_construir_features),
        ("crear_splits", cmd_crear_splits),
        ("baselines", cmd_baselines),
    ]

    for name, func in steps:
        logger.info(f"\n--- {name} ---")
        if func(args) != 0:
            logger.error(f"Fallo en {name}")
            return 1

    # Tune y train para cada modelo
    for model_type in ["mlp", "gru"]:
        args.model = model_type
        logger.info(f"\n--- tune {model_type} ---")
        if cmd_tune(args) != 0:
            return 1
        logger.info(f"\n--- entrenar {model_type} ---")
        if cmd_entrenar(args) != 0:
            return 1

    # Evaluar
    logger.info("\n--- evaluar ---")
    if cmd_evaluar(args) != 0:
        return 1

    logger.info("=== PIPELINE COMPLETO FINALIZADO ===")
    return 0


def main():
    parser = argparse.ArgumentParser(description="P1 - Pronóstico de demanda TranSmart")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preparar datos
    subparsers.add_parser("preparar-datos", help="Carga y valida dataset crudo")

    # Construir series
    subparsers.add_parser("construir-series", help="Construye series diarias por ruta")

    # Features
    subparsers.add_parser("construir-features", help="Agrega calendario y crea ventanas")

    # Splits
    subparsers.add_parser("crear-splits", help="Crea particiones temporales")

    # Baselines
    subparsers.add_parser("baselines", help="Evalúa modelos de referencia")

    # Tune
    p_tune = subparsers.add_parser("tune", help="Búsqueda de hiperparámetros")
    p_tune.add_argument("--model", choices=["mlp", "gru"], default="mlp")
    p_tune.add_argument("--trials", type=int)

    # Entrenar
    p_train = subparsers.add_parser("entrenar", help="Entrena con 30 semillas")
    p_train.add_argument("--model", choices=["mlp", "gru"], default="mlp")

    # Evaluar
    subparsers.add_parser("evaluar", help="Evalúa modelos en test")

    # Predecir (inferencia)
    p_pred = subparsers.add_parser("predecir", help="Genera pronóstico para una ruta")
    p_pred.add_argument("--ruta", required=True, help="ID ruta (ej. 11001-05001)")
    p_pred.add_argument("--fecha-origen", required=True, help="Fecha corte YYYY-MM-DD")
    p_pred.add_argument("--horizon", type=int, default=30)
    p_pred.add_argument("--quantile", type=float, default=0.90)
    p_pred.add_argument("--model", choices=["mlp", "gru"], default="gru")

    # Demo
    subparsers.add_parser("demo", help="Modo demo rápido (< 120s)")

    # Full pipeline
    subparsers.add_parser("full", help="Ejecuta pipeline completo")

    args = parser.parse_args()
    setup_logging(getattr(args, "log_level", "INFO"))

    commands = {
        "preparar-datos": cmd_preparar_datos,
        "construir-series": cmd_construir_series,
        "construir-features": cmd_construir_features,
        "crear-splits": cmd_crear_splits,
        "baselines": cmd_baselines,
        "tune": cmd_tune,
        "entrenar": cmd_entrenar,
        "evaluar": cmd_evaluar,
        "predecir": cmd_predecir,
        "demo": cmd_demo,
        "full": cmd_full,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())