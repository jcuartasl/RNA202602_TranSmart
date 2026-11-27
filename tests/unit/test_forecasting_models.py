"""Tests unitarios para modelos neuronales (transport_ai.forecasting.models)."""

import torch
import numpy as np
import pytest

from transport_ai.forecasting.models import (
    MLPForecaster,
    GRUForecaster,
    CNN1DForecaster,
    poisson_nll_loss,
    pinball_loss_torch,
    combined_loss,
    create_model,
)


class TestMLPForecaster:
    def test_forward_shape(self):
        model = MLPForecaster(input_size=60, hidden_layers=[64, 32], output_size=30)
        x = torch.randn(16, 30, 2)  # batch=16, window=30, features=2 -> input_size=60
        y = model(x)
        assert y.shape == (16, 30)

    def test_forward_2d_input(self):
        model = MLPForecaster(input_size=60, output_size=30)
        x = torch.randn(16, 60)  # ya aplanado
        y = model(x)
        assert y.shape == (16, 30)

    def test_num_params(self):
        model = MLPForecaster(input_size=60, hidden_layers=[64, 32], output_size=30)
        n_params = model.get_num_params()
        assert n_params > 0
        assert isinstance(n_params, int)


class TestGRUForecaster:
    def test_forward_shape(self):
        model = GRUForecaster(input_size=5, hidden_size=32, num_layers=2, output_size=30)
        x = torch.randn(16, 30, 5)  # batch=16, seq_len=30, features=5
        y = model(x)
        assert y.shape == (16, 30)

    def test_bidirectional(self):
        model = GRUForecaster(input_size=5, hidden_size=32, bidirectional=True, output_size=30)
        x = torch.randn(8, 20, 5)
        y = model(x)
        assert y.shape == (8, 30)

    def test_num_params(self):
        model = GRUForecaster(input_size=5, hidden_size=32, output_size=30)
        n_params = model.get_num_params()
        assert n_params > 0


class TestCNN1DForecaster:
    def test_forward_shape(self):
        model = CNN1DForecaster(input_size=5, window_size=30, output_size=30)
        x = torch.randn(16, 30, 5)
        y = model(x)
        assert y.shape == (16, 30)


class TestLosses:
    def test_poisson_nll(self):
        y_pred = torch.tensor([10.0, 20.0, 5.0, 30.0])
        y_true = torch.tensor([12.0, 18.0, 7.0, 33.0])
        loss = poisson_nll_loss(y_pred, y_true)
        # Poisson NLL simplificado (sin log(y!)) puede ser negativo
        assert torch.isfinite(loss)

    def test_poisson_nll_ceros(self):
        y_pred = torch.tensor([0.1, 5.0, 10.0])  # predicciones positivas
        y_true = torch.tensor([0.0, 5.0, 10.0])   # ceros en true
        loss = poisson_nll_loss(y_pred, y_true)
        assert torch.isfinite(loss)

    def test_pinball_loss(self):
        y_pred = torch.tensor([12.0, 25.0, 35.0])  # P90 predicho
        y_true = torch.tensor([10.0, 20.0, 30.0])
        loss = pinball_loss_torch(y_pred, y_true, quantile=0.90)
        assert loss.item() >= 0

    def test_combined_loss(self):
        y_pred_point = torch.tensor([10.0, 20.0, 30.0])
        y_pred_quant = torch.tensor([12.0, 25.0, 35.0])
        y_true = torch.tensor([11.0, 19.0, 31.0])
        loss = combined_loss(y_pred_point, y_pred_quant, y_true, quantile=0.90)
        # Combined loss puede ser negativo (Poisson NLL sin constante)
        assert torch.isfinite(loss)


class TestCreateModel:
    def test_create_mlp(self):
        config = {"input_size": 60, "hidden_layers": [64, 32], "output_size": 30}
        model = create_model("mlp", config)
        assert isinstance(model, MLPForecaster)

    def test_create_gru(self):
        config = {"input_size": 5, "hidden_size": 32, "output_size": 30}
        model = create_model("gru", config)
        assert isinstance(model, GRUForecaster)

    def test_create_cnn1d(self):
        config = {"input_size": 5, "window_size": 30, "output_size": 30}
        model = create_model("cnn1d", config)
        assert isinstance(model, CNN1DForecaster)

    def test_unknown_model(self):
        config = {"input_size": 10}
        with pytest.raises(ValueError):
            create_model("unknown", config)