"""Configuración de logging y utilidades para métricas."""

import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(
    level: int = logging.INFO,
    log_file: Path = None,
    format_str: str = None,
) -> logging.Logger:
    """
    Configura logging para consola y opcionalmente archivo.

    Args:
        level: Nivel de logging
        log_file: Ruta opcional para archivo de log
        format_str: Formato personalizado

    Returns:
        Logger configurado
    """
    if format_str is None:
        format_str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True,  # Sobrescribe configuración previa
    )

    return logging.getLogger("transmart")


def log_metrics(logger: logging.Logger, prefix: str, metrics: dict, step: int = None) -> None:
    """
    Loggea métricas de forma legible.

    Args:
        logger: Logger a usar
        prefix: Prefijo (ej. "train", "val", "test")
        metrics: Diccionario de métricas
        step: Paso/época opcional
    """
    step_str = f" [step {step}]" if step is not None else ""
    metrics_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    logger.info(f"{prefix}{step_str} | {metrics_str}")


def get_logger(name: str = "transmart") -> logging.Logger:
    """Obtiene logger con nombre específico."""
    return logging.getLogger(name)