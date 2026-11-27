"""Referencias M1: ingenuo estacional (m=7) y regresión lineal Ridge con rezagos + calendario.

Ambas predicen exactamente los mismos orígenes/horizontes que las redes. Sus cuantiles salen de los
residuos de VALIDACIÓN por horizonte (calibración tipo conformal), nunca de test.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def naive_estacional(Y: np.ndarray, route: np.ndarray, t0: np.ndarray, horizon: int, m: int = 7) -> np.ndarray:
    """ŷ(t0+h) = y(t0+h-m·ceil(h/m)): mismo día de la semana más reciente; si falta, la semana anterior."""
    h = np.arange(1, horizon + 1)
    base = t0[:, None] + h - m * np.ceil(h / m).astype(int)
    out = np.full(base.shape, np.nan)
    for k in range(4):
        out = np.where(np.isnan(out), Y[route[:, None], np.maximum(base - m * k, 0)], out)
    fallback = np.nanmean(Y[route[:, None], t0[:, None] - np.arange(28)], axis=1, keepdims=True)
    return np.where(np.isnan(out), fallback, out)


def matriz(d: dict, n_routes: int) -> np.ndarray:
    """Mismas entradas que el MLP, aplanadas: ventana + calendario futuro + one-hot de ruta."""
    return np.hstack([d["past"].reshape(len(d["y"]), -1), d["fut"].reshape(len(d["y"]), -1),
                      np.eye(n_routes, dtype="float32")[d["route"]]])


def ajustar_ridge(d: dict, n_routes: int, alpha: float) -> Ridge:
    ok = ~np.isnan(d["y"]).any(axis=1)  # Ridge no admite objetivos faltantes: se descartan esas ventanas
    return Ridge(alpha=alpha).fit(matriz(d, n_routes)[ok], d["y"][ok])


def offsets_cuantiles(y: np.ndarray, point: np.ndarray, levels: list[float]) -> np.ndarray:
    """(horizon, Q): cuantil del residuo y-ŷ por horizonte, no negativo para que P_q >= punto."""
    return np.maximum(np.nanquantile(y - point, levels, axis=0).T, 0)


def con_cuantiles(point: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return point[..., None] + np.maximum.accumulate(offsets, axis=-1)[None]
