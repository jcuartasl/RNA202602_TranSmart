"""Métricas M1 sobre la tabla larga (una fila por ruta × origen × horizonte), en pasajeros/día.

Solo se evalúan días OBSERVADOS. MASE usa el denominador estacional (m=7) calculado únicamente con el
periodo de entrenamiento de cada ruta; el agregado es la media simple entre rutas (cada ruta pesa igual).
P2 puede extender este módulo (costos, escenarios) sin cambiar la tabla larga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENTOS = [(1, 7), (8, 14), (15, 30)]


def denominador_mase(Y: np.ndarray, fechas: pd.DatetimeIndex, inicio: str, fin: str, m: int = 7) -> np.ndarray:
    """mean|y_t - y_{t-m}| por ruta dentro de [inicio, fin]; NaN si es 0 (MASE no definido)."""
    sel = (fechas >= pd.Timestamp(inicio)) & (fechas <= pd.Timestamp(fin))
    y = Y[:, sel]
    d = np.nanmean(np.abs(y[:, m:] - y[:, :-m]), axis=1)
    return np.where(d > 0, d, np.nan)


def tabla_larga(d: dict, point: np.ndarray, q: np.ndarray, levels: list[float], rutas: list[str],
                fechas: pd.DatetimeIndex, scale: np.ndarray, modelo: str) -> pd.DataFrame:
    """Desescala y aplana. Columnas: model, route_id, origin_date, horizon, date, y, point, p90 (+ otros cuantiles)."""
    n, H = d["y"].shape
    s = scale[d["route"]][:, None]
    out = pd.DataFrame({
        "model": modelo,
        "route_id": np.repeat(np.asarray(rutas)[d["route"]], H),
        "origin_date": np.repeat(fechas[d["t0"]], H),
        "horizon": np.tile(np.arange(1, H + 1), n),
        "y": (d["y"] * s).ravel(),
        "point": (point * s).ravel(),
    })
    out["date"] = out["origin_date"] + pd.to_timedelta(out["horizon"], unit="D")
    for k, lv in enumerate(levels):
        out[f"p{round(lv * 100)}"] = (q[..., k] * s).ravel()
    return out


def metricas(df: pd.DataFrame, denom: dict[str, float], q: str = "p90", tau: float = 0.9) -> pd.Series:
    df = df.dropna(subset=["y"])
    e = df["y"] - df["point"]
    eq = df["y"] - df[q]
    mae_r = e.abs().groupby(df["route_id"]).mean()
    return pd.Series({
        "mae": e.abs().mean(),
        "rmse": np.sqrt((e ** 2).mean()),
        "mase": (mae_r / mae_r.index.map(denom)).mean(),  # NaN de rutas con denominador 0 se omiten
        "pinball90": np.maximum(tau * eq, (tau - 1) * eq).mean(),
        "cobertura90": (df["y"] <= df[q]).mean(),
        "n": len(df),
    })


def resumen(df: pd.DataFrame, denom: dict[str, float], por: str | list[str] = "model") -> pd.DataFrame:
    return df.groupby(por).apply(lambda g: metricas(g, denom), include_groups=False)


def con_segmento(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["segmento"] = pd.cut(df["horizon"], [0, 7, 14, 30], labels=["1-7", "8-14", "15-30"])
    return df
