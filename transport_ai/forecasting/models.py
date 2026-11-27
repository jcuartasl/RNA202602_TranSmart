"""Redes M1: MLP y GRU, ambas multi-salida directa (30 días) con cabeza de cuantiles monotónica.

Salida por día h: punto λ_h = softplus(o_h0) ≥ 0 y cuantiles q_hk = λ_h + Σ_{j≤k} softplus(o_hj),
así P75 ≤ P80 ≤ ... ≤ P95 y todos ≥ punto por construcción (sin cruce de cuantiles).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def salida(o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    point = F.softplus(o[..., 0])
    return point, point[..., None] + torch.cumsum(F.softplus(o[..., 1:]), dim=-1)


def capas(d_in: int, hidden: list[int], d_out: int, dropout: float) -> nn.Sequential:
    layers = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    return nn.Sequential(*layers, nn.Linear(d_in, d_out))


class MLP(nn.Module):
    """Entrada aplanada: ventana (window×(2+F)) + calendario futuro (H×F) + one-hot de ruta."""

    def __init__(self, window, n_feat, n_cal, horizon, n_routes, n_q, hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.horizon, self.n_routes = horizon, n_routes
        self.net = capas(window * n_feat + horizon * n_cal + n_routes, list(hidden), horizon * (1 + n_q), dropout)

    def forward(self, past, fut, route):
        x = torch.cat([past.flatten(1), fut.flatten(1), F.one_hot(route, self.n_routes).float()], 1)
        return salida(self.net(x).view(len(x), self.horizon, -1))


class GRU(nn.Module):
    """Codificador GRU sobre la ventana; el último estado oculto + calendario futuro + ruta van a una cabeza densa.

    z_t = σ(W_z x_t + U_z h_{t-1} + b_z);  r_t = σ(W_r x_t + U_r h_{t-1} + b_r)
    h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h);  h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    """

    def __init__(self, window, n_feat, n_cal, horizon, n_routes, n_q, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        self.horizon, self.n_routes = horizon, n_routes
        self.gru = nn.GRU(n_feat, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = capas(hidden + horizon * n_cal + n_routes, [2 * hidden], horizon * (1 + n_q), dropout)

    def forward(self, past, fut, route):
        _, h = self.gru(past)
        x = torch.cat([h[-1], fut.flatten(1), F.one_hot(route, self.n_routes).float()], 1)
        return salida(self.head(x).view(len(x), self.horizon, -1))


def perdida(point, q, y, levels: torch.Tensor, kind: str = "mae") -> torch.Tensor:
    """Pérdida enmascarada: puntual (MAE o Poisson NLL sin log y!) + Σ pinball por cuantil; días desconocidos no cuentan."""
    m = ~torch.isnan(y)
    y = torch.nan_to_num(y)
    lp = (point - y).abs() if kind == "mae" else point - y * torch.log(point + 1e-6)
    e = y[..., None] - q
    lq = torch.maximum(levels * e, (levels - 1) * e).sum(-1)
    return ((lp + lq) * m).sum() / m.sum().clamp(min=1)


def crear(arch: str, **kw) -> nn.Module:
    return {"mlp": MLP, "gru": GRU}[arch](**kw)
