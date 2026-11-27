"""Tests unitarios para inferencia (transport_ai.forecasting.infer)."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from transport_ai.forecasting.infer import (
    forecast,
    forecast_batch,
    listar_rutas_disponibles,
    obtener_metadatos_modelo,
    limpiar_cache,
)


class TestInferContract:
    """Tests del contrato forecast() para P2/P5."""

    def test_forecast_retorna_dataframe_correcto(self):
        """Verifica que forecast() retorna DataFrame con columnas exactas."""
        # Mock del modelo y datos
        with patch("transport_ai.forecasting.infer._cargar_modelo_y_scaler") as mock_load, \
             patch("transport_ai.forecasting.infer._preparar_ventana_inferencia") as mock_prep, \
             patch("transport_ai.forecasting.infer.cargar_config") as mock_config:

            # Config mock
            mock_config.return_value = {
                "features": {"window_size": 30, "horizon": 30},
                "inference": {"default_model": "gru"},
            }

            # Modelo mock - retornar tensor con shape correcto
            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_output = MagicMock()
            mock_output.cpu.return_value.numpy.return_value = np.array([100.0] * 30)
            mock_model.return_value = mock_output

            mock_load.return_value = (mock_model, {}, MagicMock(), {"version": "1.0.0"})

            # Ventana mock - X con shape correcto
            mock_prep.return_value = (np.random.rand(1, 30, 5), {"feature_names": ["f1"]*5})

            # Scaler mock
            mock_scaler = MagicMock()
            mock_scaler.transform.return_value = np.random.rand(1, 150)

            # Ejecutar
            result = forecast("11001-05001", "2025-07-01", horizon=30, quantile=0.90)

            # Verificar estructura
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 30
            assert list(result.columns) == [
                "date", "route_id", "point_forecast", "quantile_forecast",
                "quantile", "origin_date", "model_version", "model_type"
            ]
            assert result["route_id"].iloc[0] == "11001-05001"
            assert result["origin_date"].iloc[0] == "2025-07-01"
            assert result["quantile"].iloc[0] == 0.90
            assert all(result["point_forecast"] >= 0)
            assert all(result["quantile_forecast"] >= result["point_forecast"])

    def test_forecast_fechas_correctas(self):
        """Verifica que las fechas empiezan al día siguiente de fecha_origen."""
        with patch("transport_ai.forecasting.infer._cargar_modelo_y_scaler") as mock_load, \
             patch("transport_ai.forecasting.infer._preparar_ventana_inferencia") as mock_prep, \
             patch("transport_ai.forecasting.infer.cargar_config") as mock_config:

            mock_config.return_value = {
                "features": {"window_size": 30, "horizon": 30},
                "inference": {"default_model": "gru"},
            }

            mock_model = MagicMock()
            mock_model.eval = MagicMock()
            mock_output = MagicMock()
            mock_output.cpu.return_value.numpy.return_value = np.array([50.0] * 5)
            mock_model.return_value = mock_output

            mock_load.return_value = (mock_model, {}, MagicMock(), {"version": "1.0.0"})
            mock_prep.return_value = (np.random.rand(1, 30, 5), {})

            result = forecast("11001-05001", "2025-07-01", horizon=5)

            # Primera fecha debe ser 2025-07-02
            assert result["date"].iloc[0].strftime("%Y-%m-%d") == "2025-07-02"
            # Última fecha debe ser 2025-07-06
            assert result["date"].iloc[-1].strftime("%Y-%m-%d") == "2025-07-06"

    def test_forecast_valida_horizon(self):
        """Valida que horizon >= 1."""
        with pytest.raises(ValueError, match="horizon debe ser >= 1"):
            forecast("11001-05001", "2025-07-01", horizon=0)

    def test_forecast_valida_quantile(self):
        """Valida que quantile esté en rango [0.50, 0.99]."""
        with pytest.raises(ValueError, match="quantile debe estar entre 0.50 y 0.99"):
            forecast("11001-05001", "2025-07-01", quantile=0.30)

        with pytest.raises(ValueError, match="quantile debe estar entre 0.50 y 0.99"):
            forecast("11001-05001", "2025-07-01", quantile=1.0)

    def test_forecast_batch_multiples_rutas(self):
        """Verifica forecast_batch procesa múltiples rutas."""
        with patch("transport_ai.forecasting.infer.forecast") as mock_forecast:
            # Mock que retorna route_id correcto para cada llamada
            def mock_forecast_fn(ruta, *args, **kwargs):
                return pd.DataFrame({
                    "date": pd.date_range("2025-07-02", periods=30),
                    "route_id": ruta,
                    "point_forecast": [100]*30,
                    "quantile_forecast": [120]*30,
                    "quantile": 0.90,
                    "origin_date": "2025-07-01",
                    "model_version": "1.0.0",
                    "model_type": "gru",
                })
            mock_forecast.side_effect = mock_forecast_fn

            result = forecast_batch(["11001-05001", "11001-76001"], "2025-07-01")

            assert len(result) == 60  # 30 * 2 rutas
            assert set(result["route_id"].unique()) == {"11001-05001", "11001-76001"}
            assert mock_forecast.call_count == 2


class TestUtilidades:
    def test_listar_rutas(self):
        """Verifica listar_rutas_disponibles."""
        with patch("transport_ai.forecasting.infer.cargar_config") as mock_config:
            mock_config.return_value = {
                "data": {"routes": ["11001-05001", "11001-76001"]}
            }
            rutas = listar_rutas_disponibles()
            assert rutas == ["11001-05001", "11001-76001"]

    def test_obtener_metadatos(self):
        """Verifica obtener_metadatos_modelo."""
        with patch("transport_ai.forecasting.infer._cargar_modelo_y_scaler") as mock_load, \
             patch("transport_ai.forecasting.infer.cargar_config") as mock_config:

            mock_config.return_value = {
                "inference": {"default_model": "gru"},
                "models": {"gru": {"hidden_size": 64}},
            }
            mock_load.return_value = (None, {}, None, {"version": "1.0.0", "deploy_seed": 42, "n_params": 12345})

            meta = obtener_metadatos_modelo("gru")
            assert meta["model_type"] == "gru"
            assert meta["version"] == "1.0.0"
            assert meta["deploy_seed"] == 42
            assert meta["n_params"] == 12345

    def test_limpiar_cache(self):
        """Verifica limpiar_cache."""
        # Agregar algo al cache
        import transport_ai.forecasting.infer as infer_module
        infer_module._MODEL_CACHE["test"] = "algo"
        assert len(infer_module._MODEL_CACHE) > 0

        limpiar_cache()
        assert len(infer_module._MODEL_CACHE) == 0