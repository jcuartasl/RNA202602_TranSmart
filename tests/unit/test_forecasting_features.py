"""Tests unitarios para features de forecasting (transport_ai.forecasting.features)."""

import pandas as pd
import numpy as np
import pytest
from datetime import date

from transport_ai.forecasting.features import (
    calcular_semana_santa,
    generar_calendario_anual,
    construir_calendario_completo,
    construir_series_diarias,
    detectar_anomalias,
    agregar_features_calendario,
    crear_ventanas_deslizantes,
)


class TestCalendario:
    def test_calcular_semana_santa(self):
        # 2024: Jueves 28 Mar, Viernes 29 Mar
        jueves, viernes = calcular_semana_santa(2024)
        assert jueves == date(2024, 3, 28)
        assert viernes == date(2024, 3, 29)

        # 2025: Jueves 17 Abr, Viernes 18 Abr
        jueves, viernes = calcular_semana_santa(2025)
        assert jueves == date(2025, 4, 17)
        assert viernes == date(2025, 4, 18)

    def test_generar_calendario_anual(self):
        cal = generar_calendario_anual(2024)
        assert len(cal) == 366  # 2024 bisiesto
        assert "fecha" in cal.columns
        assert "es_festivo" in cal.columns
        assert "dia_semana" in cal.columns
        assert "semana_santa" in cal.columns
        assert "fin_ano" in cal.columns

        # Verificar festivos conocidos
        navidad = cal[cal["fecha"] == "2024-12-25"]
        assert navidad["es_festivo"].values[0] == True
        assert "Navidad" in navidad["festivo_nombre"].values[0]

    def test_construir_calendario_completo(self):
        cal = construir_calendario_completo("2024-01-01", "2024-12-31")
        assert len(cal) == 366
        assert cal["fecha"].min() == pd.Timestamp("2024-01-01")
        assert cal["fecha"].max() == pd.Timestamp("2024-12-31")


class TestSeriesDiarias:
    def test_construir_series_basico(self):
        # Datos sintéticos
        fechas = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        df = pd.DataFrame({
            "fecha": np.repeat(fechas, 2),
            "ruta_id": ["11001-05001", "11001-76001"] * 10,
            "pasajeros": np.random.randint(50, 200, 20),
        })

        series = construir_series_diarias(df, routes=["11001-05001", "11001-76001"])

        assert len(series) == 20  # 10 días * 2 rutas
        assert "pasajeros_dia" in series.columns
        assert set(series["ruta_id"].unique()) == {"11001-05001", "11001-76001"}

    def test_construir_series_missing_zero(self):
        # Datos con gaps
        df = pd.DataFrame({
            "fecha": pd.to_datetime(["2024-01-01", "2024-01-03"]),  # Falta 02
            "ruta_id": ["11001-05001", "11001-05001"],
            "pasajeros": [100, 150],
        })

        series = construir_series_diarias(
            df, routes=["11001-05001"],
            fecha_inicio="2024-01-01", fecha_fin="2024-01-05",
            missing_policy="zero"
        )

        assert len(series) == 5  # 5 días
        # Día 02 debería ser 0
        dia_02 = series[(series["fecha"] == "2024-01-02") & (series["ruta_id"] == "11001-05001")]
        assert dia_02["pasajeros_dia"].values[0] == 0


class TestAnomalias:
    def test_detectar_valor_extremo(self):
        df = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=10),
            "ruta_id": ["11001-05001"] * 10,
            "pasajeros_dia": [100, 105, 98, 5000, 102, 99, 101, 103, 100, 97],  # 5000 es extremo
        })

        result = detectar_anomalias(df, max_pasajeros_por_despacho=100)
        anomalias = result[result["anomalia"] == True]

        assert len(anomalias) >= 1
        assert anomalias.iloc[0]["tipo_anomalia"] == "valor_extremo"


class TestFeaturesCalendario:
    def test_agregar_features_calendario(self):
        series = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=10),
            "ruta_id": ["11001-05001"] * 10,
            "pasajeros_dia": np.random.randint(50, 200, 10),
        })
        cal = construir_calendario_completo("2024-01-01", "2024-01-10")

        result = agregar_features_calendario(series, cal)

        # Debe tener features cíclicas
        assert "dia_semana_sin" in result.columns
        assert "dia_semana_cos" in result.columns
        assert "dia_ano_sin" in result.columns
        assert "es_festivo_o_puente" in result.columns
        assert "es_temporada_alta" in result.columns


class TestVentanasDeslizantes:
    def test_crear_ventanas_basico(self):
        # Serie simple con features
        n_dias = 100
        series = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=n_dias),
            "ruta_id": ["11001-05001"] * n_dias,
            "pasajeros_dia": np.random.randint(50, 200, n_dias),
            "dia_semana_sin": np.sin(2 * np.pi * np.arange(n_dias) % 7 / 7),
            "dia_semana_cos": np.cos(2 * np.pi * np.arange(n_dias) % 7 / 7),
            "es_festivo_o_puente": [False] * n_dias,
        })

        X, y, rutas, metadata = crear_ventanas_deslizantes(
            series, window_size=30, horizon=30
        )

        # Debe crear muestras
        assert X.shape[0] > 0
        assert X.shape[1] == 30  # window_size
        assert X.shape[2] >= 1  # n_features
        assert y.shape[1] == 30  # horizon
        assert len(rutas) == X.shape[0]
        assert len(metadata) == X.shape[0]

    def test_ventanas_sin_fuga_temporal(self):
        """Verifica que las ventanas no usan futuro."""
        n_dias = 80
        series = pd.DataFrame({
            "fecha": pd.date_range("2024-01-01", periods=n_dias),
            "ruta_id": ["11001-05001"] * n_dias,
            "pasajeros_dia": np.arange(n_dias),  # Valores crecientes = identificables
            "dia_semana_sin": [0] * n_dias,
        })

        X, y, _, metadata = crear_ventanas_deslizantes(
            series, window_size=30, horizon=10
        )

        # Verificar que para cada ventana, target_end > window_end
        for _, row in metadata.iterrows():
            assert row["target_start"] > row["window_end"]
            assert row["target_end"] >= row["target_start"]