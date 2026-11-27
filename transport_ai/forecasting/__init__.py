"""M1 - Pronóstico de demanda (P1). Interfaz pública para P2/P5: forecast() y rutas_disponibles()."""

from transport_ai.forecasting.infer import forecast, rutas_disponibles

__all__ = ["forecast", "rutas_disponibles"]
