"""Serialización de modelos y artefactos."""

import json
import pickle
from pathlib import Path
from typing import Any
import torch
import numpy as np


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict,
    scaler: Any = None,
    metadata: dict = None,
    version: str = "1.0.0",
) -> None:
    """
    Guarda checkpoint completo del modelo.

    Incluye: pesos, config, scaler, metadatos, versión.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "scaler": scaler,
        "metadata": metadata or {},
        "version": version,
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model_class: type = None,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict, Any, dict]:
    """
    Carga checkpoint y reconstruye modelo.

    Returns:
        (model, config, scaler, metadata)
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    scaler = checkpoint.get("scaler")
    metadata = checkpoint.get("metadata", {})
    version = checkpoint.get("version", "unknown")

    # Reconstruir modelo si se proporciona la clase
    model = None
    if model_class is not None:
        model = model_class(**config.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

    return model, config, scaler, metadata


def save_scaler(scaler: Any, path: Path) -> None:
    """Guarda scaler/preprocesador con pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path) -> Any:
    """Carga scaler/preprocesador."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: dict, path: Path) -> None:
    """Guarda diccionario como JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> dict:
    """Carga JSON como diccionario."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(array: np.ndarray, path: Path) -> None:
    """Guarda array numpy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    """Carga array numpy."""
    return np.load(path)