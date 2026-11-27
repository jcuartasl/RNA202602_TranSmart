"""P1 - Módulo de Pronóstico de Demanda y Modelos Temporales.

Interfaces públicas para inferencia y utilidades.
"""

from transport_ai.forecasting.infer import (
    forecast,
    forecast_batch,
    listar_rutas_disponibles,
    obtener_metadatos_modelo,
    limpiar_cache,
)

from transport_ai.forecasting.data import (
    cargar_dataset_crudo,
    cargar_procesado,
)

from transport_ai.forecasting.features import (
    construir_series_diarias,
    agregar_features_calendario,
    crear_ventanas_deslizantes,
)

from transport_ai.forecasting.baselines import (
    naive_seasonal_forecast,
    linear_lags_forecast,
    forecast_baseline,
)

__version__ = "1.0.0"
__author__ = "P1 - Equipo TranSmart"

__all__ = [
    # Inferencia (contrato P2/P5)
    "forecast",
    "forecast_batch",
    "listar_rutas_disponibles",
    "obtener_metadatos_modelo",
    "limpiar_cache",
    # Datos
    "cargar_dataset_crudo",
    "cargar_procesado",
    # Features
    "construir_series_diarias",
    "agregar_features_calendario",
    "crear_ventanas_deslizantes",
    # Baselines
    "naive_seasonal_forecast",
    "linear_lags_forecast",
    "forecast_baseline",
]