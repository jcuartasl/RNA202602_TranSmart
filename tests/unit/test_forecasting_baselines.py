"""Tests unitarios para baselines (transport_ai.forecasting.baselines)."""

import pandas as pd
import numpy as np
import pytest
from datetime import date

from transport_ai.forecasting.baselines import (
    naive_seasonal_forecast,
    linear_lags_forecast,
    forecast_baseline,
)


class TestNaiveSeasonal:
    def test_naive_seasonal_basico(self):
        """Baseline ingenuo estacional con datos suficientes."""
        fechas = pd.date_range("2024-01-01", periods=60, freq="D")
        series = pd.DataFrame({
            "fecha": fechas,
            "ruta_id": ["11001-05001"] * 60,
            "pasajeros_dia": np.random.randint(80, 120, 60),
        })

        point, quantile = naive_seasonal_forecast(
            series, "11001-05001", "2024-03-01", horizon=30
        )

        assert len(point) == 30
        assert len(quantile) == 30
        assert all(point >= 0)
        assert all(quantile >= point)

    def test_naive_seasonal_pocos_datos(self):
        """Con pocos datos, usa media histórica."""
        series = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=3),
            "ruta_id": ["11001-05001"] * 3,
            "pasajeros_dia": [100, 110, 105],
        })

        point, quantile = naive_seasonal_forecast(
            series, "11001-05001", "2024-01-10", horizon=5
        )

        assert len(point) == 5
        assert all(p == point[0] for p in point)  # Todos iguales (media)


class TestLinearLags:
    def test_linear_lags_basico(self):
        """Baseline lineal con features de calendario."""
        n = 100
        fechas = pd.date_range("2024-01-01", periods=n, freq="D")
        series = pd.DataFrame({
            "fecha": fechas,
            "ruta_id": ["11001-05001"] * n,
            "pasajeros_dia": np.random.randint(80, 120, n),
            "dia_semana_sin": np.sin(2 * np.pi * np.arange(n) % 7 / 7),
            "dia_semana_cos": np.cos(2 * np.pi * np.arange(n) % 7 / 7),
            "es_festivo_o_puente": [False] * n,
            "es_temporada_alta": [False] * n,
            "temporada_escolar": [True] * n,
        })

        point, quantile = linear_lags_forecast(
            series, "11001-05001", "2024-04-10", horizon=10
        )

        assert len(point) == 10
        assert len(quantile) == 10
        assert all(point >= 0)

    def test_linear_lags_pocos_datos(self):
        """Con pocos datos, fallback a media."""
        series = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=5),
            "ruta_id": ["11001-05001"] * 5,
            "pasajeros_dia": [100, 110, 105, 115, 108],
        })

        point, quantile = linear_lags_forecast(
            series, "11001-05001", "2024-01-20", horizon=5
        )

        assert len(point) == 5
        assert all(p >= 0 for p in point)


class TestForecastBaseline:
    def test_forecast_baseline_formato(self):
        """Verifica formato de salida estándar."""
        fechas = pd.date_range("2024-01-01", periods=60, freq="D")
        series = pd.DataFrame({
            "fecha": fechas,
            "ruta_id": ["11001-05001"] * 60,
            "pasajeros_dia": np.random.randint(80, 120, 60),
            "dia_semana_sin": np.sin(2 * np.pi * np.arange(60) % 7 / 7),
            "dia_semana_cos": np.cos(2 * np.pi * np.arange(60) % 7 / 7),
            "es_festivo_o_puente": [False] * 60,
            "es_temporada_alta": [False] * 60,
            "temporada_escolar": [True] * 60,
        })

        result = forecast_baseline(
            series, "11001-05001", "2024-03-01",
            horizon=10, method="naive_seasonal", quantile=0.90
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert list(result.columns) == [
            "date", "route_id", "point_forecast", "quantile_forecast",
            "quantile", "model"
        ]
        assert result["route_id"].iloc[0] == "11001-05001"
        assert result["model"].iloc[0] == "baseline_naive_seasonal"
        assert result["quantile"].iloc[0] == 0.90

    def test_forecast_baseline_fechas(self):
        """Verifica fechas de predicción."""
        series = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=60, freq="D"),
            "ruta_id": ["11001-05001"] * 60,
            "pasajeros_dia": [100] * 60,
            "dia_semana_sin": [0] * 60,
            "dia_semana_cos": [1] * 60,
            "es_festivo_o_puente": [False] * 60,
        })

        result = forecast_baseline(
            series, "11001-05001", "2024-03-01", horizon=5
        )

        # Primera predicción = 2024-03-02
        assert result["date"].iloc[0].strftime("%Y-%m-%d") == "2024-03-02"
        assert result["date"].iloc[-1].strftime("%Y-%m-%d") == "2024-03-06"