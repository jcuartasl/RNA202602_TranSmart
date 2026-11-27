"""Gestión de semillas aleatorias para reproducibilidad."""

import random
import os
import numpy as np
import torch


# 30 semillas independientes: 42 a 71 inclusive
SEEDS = list(range(42, 72))
DEPLOY_SEED = 42  # Semilla fija pre-definida ANTES de ver test


def set_seed(seed: int) -> None:
    """Fija todas las semillas para reproducibilidad total."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo en CuDNN (puede afectar rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Variable de entorno para hash de Python
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_seeds(count: int = 30, start: int = 42) -> list[int]:
    """Retorna lista de semillas consecutivas."""
    return list(range(start, start + count))