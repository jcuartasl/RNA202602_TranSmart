"""Carga, validación y preparación del dataset oficial de transporte."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

from transport_ai.common.paths import FORECASTING_PROCESSED, DATA_MANIFESTS, DATA_RAW
from transport_ai.common.logging_utils import get_logger

logger = get_logger(__name__)

# Columnas esperadas según diccionario Superintendencia Transporte
ESPERADAS = [
    "fecha_despacho",
    "origen_divipola",
    "destino_divipola",
    "pasajeros_abordados",
    "vehiculos_despachados",
    "clase_vehiculo",
    "empresa_transportadora",
]

# Columnas canónicas internas
CANONICAS = {
    "fecha_despacho": "fecha",
    "origen_divipola": "origen",
    "destino_divipola": "destino",
    "pasajeros_abordados": "pasajeros",
    "vehiculos_despachados": "vehiculos",
    "clase_vehiculo": "clase",
    "empresa_transportadora": "empresa",
}


def cargar_config(path: Path = None) -> dict:
    """Carga configuración forecast.yaml."""
    if path is None:
        from transport_ai.common.paths import CONFIGS
        path = CONFIGS / "forecast.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validar_columnas(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Valida que el DataFrame tenga las columnas esperadas."""
    faltantes = [c for c in ESPERADAS if c not in df.columns]
    extra = [c for c in df.columns if c not in ESPERADAS]
    return len(faltantes) == 0, faltantes, extra


def validar_tipos(df: pd.DataFrame) -> list[str]:
    """Valida tipos de datos básicos y reporta problemas."""
    problemas = []

    if not pd.api.types.is_datetime64_any_dtype(df["fecha_despacho"]):
        problemas.append("fecha_despacho: no es datetime")

    # DIVIPOLA deben ser strings (preservar ceros iniciales)
    for col in ["origen_divipola", "destino_divipola"]:
        if not df[col].dtype == "object":
            problemas.append(f"{col}: debería ser string/object para preservar ceros iniciales")

    # Numéricas
    for col in ["pasajeros_abordados", "vehiculos_despachados"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problemas.append(f"{col}: no es numérico")

    return problemas


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas a nombres canónicos internos."""
    df = df.rename(columns=CANONICAS).copy()

    # Asegurar tipos correctos
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["origen"] = df["origen"].astype(str).str.zfill(5)
    df["destino"] = df["destino"].astype(str).str.zfill(5)
    df["pasajeros"] = pd.to_numeric(df["pasajeros"], errors="coerce").fillna(0).astype(int)
    df["vehiculos"] = pd.to_numeric(df["vehiculos"], errors="coerce").fillna(0).astype(int)

    # Identificador de ruta: origen-destino (dirigido)
    df["ruta_id"] = df["origen"] + "-" + df["destino"]

    return df


def calcular_hash_archivo(path: Path) -> str:
    """Calcula SHA256 de un archivo para trazabilidad."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generar_manifiesto(
    raw_path: Path,
    processed_dir: Path,
    routes: list[str],
    output_path: Path,
) -> dict:
    """Genera manifiesto de datos con hashes y metadatos."""
    import datetime

    hash_raw = calcular_hash_archivo(raw_path) if raw_path.exists() else "no_disponible"

    manifiesto = {
        "fecha_generacion": datetime.datetime.now().isoformat(),
        "dataset_raw": {
            "path": str(raw_path),
            "sha256": hash_raw,
        },
        "rutas_seleccionadas": routes,
        "procesados": {
            "directorio": str(processed_dir),
            "archivos": [],
        },
    }

    # Agregar archivos procesados si existen
    if processed_dir.exists():
        for archivo in processed_dir.glob("**/*"):
            if archivo.is_file():
                manifiesto["procesados"]["archivos"].append({
                    "path": str(archivo.relative_to(processed_dir)),
                    "sha256": calcular_hash_archivo(archivo),
                    "size_bytes": archivo.stat().st_size,
                })

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=2, ensure_ascii=False)

    logger.info(f"Manifiesto guardado en {output_path}")
    return manifiesto


def cargar_dataset_crudo(path: Path = None) -> pd.DataFrame:
    """
    Carga dataset crudo desde CSV.

    Args:
        path: Ruta al archivo CSV. Si None, usa config.

    Returns:
        DataFrame con columnas canónicas validadas
    """
    config = cargar_config()
    if path is None:
        path = Path(config["data"]["raw_path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en {path}. "
            f"Descargar desde {config['data']['source_url']} y colocar en data/raw/"
        )

    logger.info(f"Cargando dataset desde {path}")
    df = pd.read_csv(path, low_memory=False)

    # Validar columnas
    ok, faltantes, extra = validar_columnas(df)
    if not ok:
        raise ValueError(f"Columnas faltantes: {faltantes}. Columnas extra: {extra}")

    # Validar tipos
    problemas = validar_tipos(df)
    if problemas:
        logger.warning(f"Problemas de tipos detectados: {problemas}")

    # Normalizar
    df = normalizar_columnas(df)

    logger.info(f"Dataset cargado: {len(df)} registros, {df['ruta_id'].nunique()} rutas")
    logger.info(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")

    return df


def guardar_procesado(df: pd.DataFrame, nombre: str, directorio: Path = None) -> Path:
    """Guarda DataFrame procesado en formato Parquet."""
    if directorio is None:
        directorio = FORECASTING_PROCESSED
    directorio.mkdir(parents=True, exist_ok=True)

    path = directorio / f"{nombre}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Guardado {nombre}: {len(df)} filas en {path}")
    return path


def cargar_procesado(nombre: str, directorio: Path = None) -> pd.DataFrame:
    """Carga DataFrame procesado desde Parquet."""
    if directorio is None:
        directorio = FORECASTING_PROCESSED
    path = directorio / f"{nombre}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Archivo procesado no encontrado: {path}")
    return pd.read_parquet(path)