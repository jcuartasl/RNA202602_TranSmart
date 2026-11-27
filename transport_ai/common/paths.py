"""Rutas canónicas del proyecto."""

from pathlib import Path

# Raíz del proyecto (directorio que contiene transport_ai/)
ROOT = Path(__file__).resolve().parents[2]

# Directorios de datos
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_MANIFESTS = ROOT / "data" / "manifests"

# Subdirectorios por módulo
FORECASTING_PROCESSED = DATA_PROCESSED / "forecasting"
RECOMMENDER_PROCESSED = DATA_PROCESSED / "recommender"
VISION_PROCESSED = DATA_PROCESSED / "vision"

# Artefactos entrenados
ARTIFACTS = ROOT / "artifacts"
FORECASTING_ARTIFACTS = ARTIFACTS / "forecasting"
RECOMMENDER_ARTIFACTS = ARTIFACTS / "recommender"
VISION_ARTIFACTS = ARTIFACTS / "vision"

# Configuraciones
CONFIGS = ROOT / "configs"

# Experimentos y reportes
EXPERIMENTS = ROOT / "experiments"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

# App
APP_DIR = ROOT / "app"


def ensure_dirs() -> None:
    """Crea todos los directorios necesarios si no existen."""
    for path in [
        DATA_RAW,
        FORECASTING_PROCESSED,
        RECOMMENDER_PROCESSED,
        VISION_PROCESSED,
        DATA_MANIFESTS,
        FORECASTING_ARTIFACTS,
        RECOMMENDER_ARTIFACTS,
        VISION_ARTIFACTS,
        CONFIGS,
        EXPERIMENTS,
        FIGURES,
        TABLES,
    ]:
        path.mkdir(parents=True, exist_ok=True)