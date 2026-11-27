"""Tests unitarios para métricas de forecasting (transport_ai.common.metrics)."""

import numpy as np
import pytest

from transport_ai.common.metrics import (
    mae,
    rmse,
    mase,
    pinball_loss,
    coverage,
    evaluate_forecast,
    evaluate_by_horizon,
)


class TestMAE:
    def test_mae_basico(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        assert abs(mae(y_true, y_pred) - 2.333) < 0.01

    def test_mae_cero(self):
        y_true = np.array([10, 10, 10])
        y_pred = np.array([10, 10, 10])
        assert mae(y_true, y_pred) == 0.0


class TestRMSE:
    def test_rmse_basico(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        # RMSE = sqrt(mean((y_true - y_pred)^2)) = sqrt((4+4+9)/3) = sqrt(17/3) ≈ 2.380
        assert abs(rmse(y_true, y_pred) - 2.380) < 0.01


class TestMASE:
    def test_mase_normal(self):
        y_train = np.array([10, 12, 11, 13, 12, 14, 13, 15, 14, 16])  # tendencia suave
        y_true = np.array([15, 17])
        y_pred = np.array([14, 18])
        # MASE debería ser razonable
        score = mase(y_true, y_pred, y_train, seasonal_period=7)
        assert score > 0
        assert np.isfinite(score)

    def test_mase_serie_constante(self):
        y_train = np.array([10, 10, 10, 10, 10])
        y_true = np.array([10, 10])
        y_pred = np.array([10, 10])
        # MASE = inf cuando serie constante en train (denominador = 0)
        score = mase(y_true, y_pred, y_train, seasonal_period=7)
        assert score == float("inf")

    def test_mase_serie_constante_error(self):
        y_train = np.array([10, 10, 10, 10, 10])
        y_true = np.array([12, 12])
        y_pred = np.array([10, 10])
        # Serie constante en train -> denominador = 0 -> inf
        score = mase(y_true, y_pred, y_train, seasonal_period=7)
        assert score == float("inf")


class TestPinballLoss:
    def test_pinball_p90(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 25, 35])  # Cuantil P90 predicho
        # Para P90, pérdida penaliza más subestimación
        loss = pinball_loss(y_true, y_pred, quantile=0.90)
        assert loss >= 0

    def test_pinball_perfecto(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        loss = pinball_loss(y_true, y_pred, quantile=0.90)
        assert loss == 0.0


class TestCoverage:
    def test_cobertura_p90(self):
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred_upper = np.array([15, 25, 35, 45, 55])  # Todos cubren
        assert coverage(y_true, y_pred_upper) == 1.0

    def test_cobertura_parcial(self):
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred_upper = np.array([5, 25, 35, 45, 55])  # Solo 4 de 5 cubren
        assert abs(coverage(y_true, y_pred_upper) - 0.8) < 0.01


class TestEvaluateForecast:
    def test_evaluate_completo(self):
        y_train = np.array([10, 12, 11, 13, 12, 14, 13, 15, 14, 16] * 3)  # 30 días
        y_true = np.array([15, 17, 16, 18])
        y_pred = np.array([14, 18, 15, 19])
        y_quant = np.array([16, 20, 17, 21])

        metrics = evaluate_forecast(y_true, y_pred, y_quant, y_train, quantile=0.90)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mase" in metrics
        assert "pinball" in metrics
        assert "coverage" in metrics
        assert all(np.isfinite(v) for v in metrics.values())


class TestEvaluateByHorizon:
    def test_por_tramos(self):
        y_train = np.array([10] * 30)
        y_true = np.random.rand(30) * 10 + 10
        y_pred = y_true + np.random.randn(30) * 2
        y_quant = y_pred * 1.2

        segments = [(1, 7), (8, 14), (15, 30)]
        results = evaluate_by_horizon(y_true, y_pred, y_quant, y_train, segments)

        assert "h1_7" in results
        assert "h8_14" in results
        assert "h15_30" in results
        for seg_metrics in results.values():
            assert "mae" in seg_metrics
            assert "rmse" in seg_metrics