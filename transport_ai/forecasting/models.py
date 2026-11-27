"""Modelos neuronales: MLP y GRU para pronóstico de series temporales."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPForecaster(nn.Module):
    """
    MLP (Perceptrón Multicapa) para pronóstico multi-output directo.

    Arquitectura:
    - Input: ventana de rezagos (window_size * n_features) aplanada
    - Hidden layers: configurables con dropout y batch norm opcional
    - Output: horizon (30 días) - predicción directa multi-output

    Ventaja: sin recurrencia, entrena rápido, captura patrones no lineales.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int] = None,
        output_size: int = 30,
        dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        # Función de activación
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Activación no soportada: {activation}")

        # Construir capas
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Inicialización
        self._init_weights()

    def _init_weights(self):
        """Inicialización Xavier para capas lineales."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, window_size, n_features) o (batch_size, input_size)

        Returns:
            (batch_size, output_size) - pronóstico para 30 días
        """
        # Aplanar si viene en 3D
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def get_num_params(self) -> int:
        """Retorna número de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRUForecaster(nn.Module):
    """
    GRU (Gated Recurrent Unit) para pronóstico multi-output directo.

    Arquitectura:
    - GRU layers: procesa secuencia temporal (window_size, n_features)
    - Toma último hidden state (o todos) y proyecta a horizon
    - Multi-output directo: 30 salidas simultáneas

    Ventaja: captura dependencias temporales largas, menos parámetros que LSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 30,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Factor de dirección
        dir_factor = 2 if bidirectional else 1

        # Proyección a horizonte
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * dir_factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        self._init_weights()

    def _init_weights(self):
        """Inicialización ortogonal para GRU, Xavier para proyección."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, window_size, n_features)

        Returns:
            (batch_size, output_size) - pronóstico para 30 días
        """
        # GRU espera (batch, seq_len, input_size)
        if x.dim() == 2:
            # Si viene aplanado, reconstruir (asumir window_size conocido)
            # Esto no debería pasar en uso normal
            raise ValueError("GRU espera input 3D: (batch, window_size, n_features)")

        # Forward GRU
        # output: (batch, seq_len, hidden_size * dir_factor)
        # h_n: (num_layers * dir_factor, batch, hidden_size)
        output, h_n = self.gru(x)

        # Usar último hidden state de la última capa
        if self.bidirectional:
            # Concatenar forward y backward de última capa
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        else:
            h_last = h_n[-1]  # (batch, hidden_size)

        # Proyectar a horizonte
        return self.output_proj(h_last)

    def get_num_params(self) -> int:
        """Retorna número de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN1DForecaster(nn.Module):
    """
    CNN 1D para pronóstico multi-output directo (alternativa a GRU/LSTM).

    Arquitectura:
    - Conv1D layers: extraen features temporales locales
    - Global pooling + Dense layers: proyectan a horizonte

    Ventaja: muy rápida, captura patrones locales, menos parámetros.
    """

    def __init__(
        self,
        input_size: int,  # n_features
        window_size: int,  # seq_len
        output_size: int = 30,
        channels: list[int] = None,
        kernel_sizes: list[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 128, 64]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        self.input_size = input_size
        self.window_size = window_size
        self.output_size = output_size

        # CNN 1D: (batch, channels, seq_len)
        conv_layers = []
        in_channels = input_size

        for out_channels, kernel_size in zip(channels, kernel_sizes):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Global average pooling + proyección
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], output_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, window_size, n_features)

        Returns:
            (batch_size, output_size)
        """
        # Transpose para Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (batch, channels)
        return self.output_proj(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Funciones de pérdida para conteos y cuantiles

def poisson_nll_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Poisson Negative Log Likelihood para variable de conteo.

    L = y_pred - y_true * log(y_pred + eps)  (ignorando log(y_true!) que es constante)

    Args:
        y_pred: Predicciones (rate parameter lambda), debe ser > 0
        y_true: Valores reales (conteos enteros >= 0)
        eps: Pequeño epsilon para estabilidad numérica

    Returns:
        Loss escalar
    """
    # Asegurar predicciones positivas
    y_pred = torch.clamp(y_pred, min=eps)
    loss = y_pred - y_true * torch.log(y_pred)
    return loss.mean()


def pinball_loss_torch(y_pred: torch.Tensor, y_true: torch.Tensor, quantile: float = 0.90) -> torch.Tensor:
    """
    Pinball loss (quantile loss) en PyTorch.

    L_q(y, q) = max(q * (y - q_pred), (q - 1) * (y - q_pred))

    Args:
        y_pred: Cuantil predicho
        y_true: Valor real
        quantile: Cuantil objetivo (0.90 para P90)

    Returns:
        Loss escalar
    """
    error = y_true - y_pred
    loss = torch.max(quantile * error, (quantile - 1) * error)
    return loss.mean()


def combined_loss(
    y_pred_point: torch.Tensor,
    y_pred_quantile: torch.Tensor,
    y_true: torch.Tensor,
    quantile: float = 0.90,
    point_weight: float = 1.0,
    quantile_weight: float = 0.5,
    point_loss_type: str = "poisson",
) -> torch.Tensor:
    """
    Pérdida combinada: punto + cuantil.

    Args:
        y_pred_point: Pronóstico puntual (batch, horizon)
        y_pred_quantile: Pronóstico cuantil P90 (batch, horizon)
        y_true: Valores reales (batch, horizon)
        quantile: Cuantil objetivo
        point_weight: Peso pérdida puntual
        quantile_weight: Peso pérdida cuantil
        point_loss_type: "poisson", "mse", "mae"

    Returns:
        Loss combinado escalar
    """
    # Pérdida puntual
    if point_loss_type == "poisson":
        loss_point = poisson_nll_loss(y_pred_point, y_true)
    elif point_loss_type == "mse":
        loss_point = F.mse_loss(y_pred_point, y_true)
    elif point_loss_type == "mae":
        loss_point = F.l1_loss(y_pred_point, y_true)
    else:
        raise ValueError(f"point_loss_type no soportado: {point_loss_type}")

    # Pérdida cuantil (pinball)
    loss_quantile = pinball_loss_torch(y_pred_quantile, y_true, quantile)

    return point_weight * loss_point + quantile_weight * loss_quantile


def create_model(model_type: str, config: dict) -> nn.Module:
    """
    Factory para crear modelos según configuración.

    Args:
        model_type: "mlp", "gru", "cnn1d"
        config: Diccionario con hiperparámetros

    Returns:
        Modelo instanciado
    """
    if model_type == "mlp":
        return MLPForecaster(
            input_size=config["input_size"],
            hidden_layers=config.get("hidden_layers", [128, 64, 32]),
            output_size=config.get("output_size", 30),
            dropout=config.get("dropout", 0.2),
            activation=config.get("activation", "relu"),
            use_batch_norm=config.get("use_batch_norm", True),
        )
    elif model_type == "gru":
        return GRUForecaster(
            input_size=config["input_size"],
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            output_size=config.get("output_size", 30),
            dropout=config.get("dropout", 0.2),
            bidirectional=config.get("bidirectional", False),
        )
    elif model_type == "cnn1d":
        return CNN1DForecaster(
            input_size=config["input_size"],
            window_size=config["window_size"],
            output_size=config.get("output_size", 30),
            channels=config.get("channels", [64, 128, 64]),
            kernel_sizes=config.get("kernel_sizes", [3, 3, 3]),
            dropout=config.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")