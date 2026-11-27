"""Particiones temporales y origen rodante (rolling origin) para evaluación sin fuga."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import yaml

from transport_ai.common.paths import FORECASTING_PROCESSED
from transport_ai.common.logging_utils import get_logger

logger = get_logger(__name__)


def cargar_config() -> dict:
    """Carga configuración forecast.yaml."""
    from transport_ai.common.paths import CONFIGS
    with open(CONFIGS / "forecast.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def crear_particiones_fijas(
    metadata: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, np.ndarray]:
    """
    Crea particiones fijas train/val/test basadas en fecha_origen de las ventanas.

    Args:
        metadata: DataFrame con columna 'fecha_origen' (y opcionalmente 'ruta_id')
        train_end, val_start, val_end, test_start, test_end: Fechas corte "YYYY-MM-DD"

    Returns:
        Diccionario con índices booleanos o arrays de índices para cada split
    """
    fecha_origen = pd.to_datetime(metadata["fecha_origen"])

    train_mask = fecha_origen <= pd.Timestamp(train_end)
    val_mask = (fecha_origen >= pd.Timestamp(val_start)) & (fecha_origen <= pd.Timestamp(val_end))
    test_mask = (fecha_origen >= pd.Timestamp(test_start)) & (fecha_origen <= pd.Timestamp(test_end))

    # Verificar que no hay solapamiento
    assert not (train_mask & val_mask).any(), "Solapamiento train/val"
    assert not (val_mask & test_mask).any(), "Solapamiento val/test"
    assert not (train_mask & test_mask).any(), "Solapamiento train/test"

    # Verificar que hay datos en cada split
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        n = mask.sum()
        if n == 0:
            logger.warning(f"Split {name} vacío. Verificar fechas de corte.")
        else:
            logger.info(f"Split {name}: {n} ventanas")

    return {
        "train": np.where(train_mask)[0],
        "val": np.where(val_mask)[0],
        "test": np.where(test_mask)[0],
    }


def crear_origenes_rodantes(
    metadata: pd.DataFrame,
    rolling_origins: list[str],
    horizon: int = 30,
) -> list[dict[str, np.ndarray]]:
    """
    Crea particiones de origen rodante (rolling origin) para backtesting.

    Para cada fecha de origen en rolling_origins:
    - train: todas las ventanas con fecha_origen < origen
    - test: ventanas con fecha_origen == origen (horizonte completo)

    Args:
        metadata: DataFrame con 'fecha_origen', 'ruta_id'
        rolling_origins: Lista de fechas "YYYY-MM-DD" como orígenes de evaluación
        horizon: Horizonte de pronóstico (días)

    Returns:
        Lista de dicts con keys 'train', 'test', 'origin_date'
    """
    fecha_origen = pd.to_datetime(metadata["fecha_origen"])
    resultados = []

    for origen_str in rolling_origins:
        origen = pd.Timestamp(origen_str)

        train_mask = fecha_origen < origen
        test_mask = fecha_origen == origen

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_test == 0:
            logger.warning(f"Origen {origen_str}: sin ventanas de test (horizonte {horizon} días)")
            continue

        resultados.append({
            "origin_date": origen_str,
            "train": np.where(train_mask)[0],
            "test": np.where(test_mask)[0],
            "n_train": int(n_train),
            "n_test": int(n_test),
        })
        logger.info(f"Rolling origin {origen_str}: train={n_train}, test={n_test}")

    return resultados


def guardar_splits(
    splits_fijos: dict,
    splits_rodantes: list,
    output_path: Path,
) -> None:
    """Guarda particiones en formato JSON para reproducibilidad."""
    import json
    from datetime import datetime

    data = {
        "fecha_generacion": datetime.now().isoformat(),
        "splits_fijos": {k: v.tolist() for k, v in splits_fijos.items()},
        "splits_rodantes": [
            {**s, "train": s["train"].tolist(), "test": s["test"].tolist()}
            for s in splits_rodantes
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Particiones guardadas en {output_path}")


def cargar_splits(path: Path) -> dict:
    """Carga particiones desde JSON."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convertir listas de vuelta a arrays
    for k in data["splits_fijos"]:
        data["splits_fijos"][k] = np.array(data["splits_fijos"][k])
    for s in data["splits_rodantes"]:
        s["train"] = np.array(s["train"])
        s["test"] = np.array(s["test"])

    return data


def aplicar_splits(
    X: np.ndarray,
    y: np.ndarray,
    rutas: np.ndarray,
    metadata: pd.DataFrame,
    splits: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Aplica índices de splits a datos (X, y, rutas, metadata).

    Returns:
        Dict con keys 'train', 'val', 'test' -> (X, y, rutas, metadata)
    """
    resultado = {}
    for split_name, indices in splits.items():
        if len(indices) == 0:
            resultado[split_name] = (np.array([]), np.array([]), np.array([]), pd.DataFrame())
            continue

        resultado[split_name] = (
            X[indices],
            y[indices],
            rutas[indices],
            metadata.iloc[indices].reset_index(drop=True),
        )

    return resultado


def verificar_sin_fuga(
    splits_fijos: dict,
    splits_rodantes: list,
    metadata: pd.DataFrame,
) -> bool:
    """
    Verifica que no hay fuga temporal entre particiones.

    Condiciones:
    - train < val < test en splits fijos
    - En rolling origin: train siempre anterior a test
    - No hay solapamiento de fechas_origen entre splits
    """
    fecha_origen = pd.to_datetime(metadata["fecha_origen"])

    # Splits fijos
    train_max = fecha_origen.iloc[splits_fijos["train"]].max() if len(splits_fijos["train"]) > 0 else None
    val_min = fecha_origen.iloc[splits_fijos["val"]].min() if len(splits_fijos["val"]) > 0 else None
    val_max = fecha_origen.iloc[splits_fijos["val"]].max() if len(splits_fijos["val"]) > 0 else None
    test_min = fecha_origen.iloc[splits_fijos["test"]].min() if len(splits_fijos["test"]) > 0 else None

    ok = True
    if train_max and val_min and train_max >= val_min:
        logger.error(f"FUGA: train_max ({train_max}) >= val_min ({val_min})")
        ok = False
    if val_max and test_min and val_max >= test_min:
        logger.error(f"FUGA: val_max ({val_max}) >= test_min ({test_min})")
        ok = False

    # Rolling origins
    for s in splits_rodantes:
        train_max_ro = fecha_origen.iloc[s["train"]].max() if len(s["train"]) > 0 else None
        test_min_ro = fecha_origen.iloc[s["test"]].min() if len(s["test"]) > 0 else None
        if train_max_ro and test_min_ro and train_max_ro >= test_min_ro:
            logger.error(f"FUGA rolling origin {s['origin_date']}: train_max >= test_min")
            ok = False

    if ok:
        logger.info("Verificación anti-fuga: OK - No hay fuga temporal detectada")

    return ok