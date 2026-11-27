"""Contrato M1 para P2/P5 (sin reentrenar): forecast(route_id, origin_date, horizon=30, quantile=0.90).

origin_date = fecha de corte = último día con historia disponible; se pronostican origin_date+1 .. +horizon.
Salida: date, route_id, point, upper_quantile, origin_date, model_version, horizon, quantile, model, observed
(pasajeros/día; observed = dato real si existe en la serie procesada, NaN si es futuro desconocido).
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import torch

from transport_ai.common.paths import FORECASTING_ARTIFACTS, FORECASTING_PROCESSED
from transport_ai.forecasting import baselines as B
from transport_ai.forecasting.features import calendario, panel, ventanas

MODELOS = ("mlp", "gru", "ridge", "naive")


@lru_cache(maxsize=None)
def metadata() -> dict:
    return json.loads((FORECASTING_ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _red(arch: str):
    from transport_ai.forecasting.train import construir
    ck = torch.load(FORECASTING_ARTIFACTS / f"{arch}.pt", map_location="cpu", weights_only=True)
    model = construir(arch, ck["hp"], {"H": ck["horizon"], "rutas": ck["rutas"], "levels": ck["levels"]})
    model.load_state_dict(ck["state"])
    return model.eval(), ck


@lru_cache(maxsize=None)
def _panel():
    meta = metadata()
    Y, fechas = panel(pd.read_parquet(FORECASTING_PROCESSED / "series.parquet"), meta["rutas"])
    H = meta["horizon"]
    futuro = pd.date_range(fechas[-1] + pd.Timedelta(days=1), periods=H)  # espacio para pronosticar después del último dato
    fechas_ext = fechas.append(futuro)
    Y = np.hstack([Y, np.full((len(Y), H), np.nan)])
    return Y / np.asarray(meta["scale"])[:, None], fechas_ext, calendario(fechas_ext).to_numpy(), len(fechas)


def rutas_disponibles() -> pd.DataFrame:
    """Catálogo para selectores de la app: route_id, nombres DIVIPOLA y rango válido de origin_date."""
    return pd.DataFrame(metadata()["catalogo"])


def forecast(route_id: str, origin_date, horizon: int = 30, quantile: float = 0.90, model: str | None = None) -> pd.DataFrame:
    meta = metadata()
    model = model or meta["default_model"]
    levels = meta["levels"]
    if model not in MODELOS:
        raise ValueError(f"model debe ser uno de {MODELOS}")
    if route_id not in meta["rutas"]:
        raise ValueError(f"Ruta {route_id} no modelada. Ver rutas_disponibles()")
    if not 1 <= horizon <= meta["horizon"]:
        raise ValueError(f"horizon debe estar entre 1 y {meta['horizon']}")
    if not levels[0] <= quantile <= levels[-1]:
        raise ValueError(f"quantile debe estar en [{levels[0]}, {levels[-1]}] (cuantiles entrenados {levels})")
    Ys, fechas, cal, n_obs = _panel()
    origin = pd.Timestamp(origin_date)
    lo, hi = pd.Timestamp(meta["origin_min"]), fechas[n_obs - 1]
    if not lo <= origin <= hi:
        raise ValueError(f"origin_date debe estar entre {lo.date()} y {hi.date()} (historia disponible)")
    r, t0 = meta["rutas"].index(route_id), fechas.get_loc(origin)
    H, scale = meta["horizon"], meta["scale"][meta["rutas"].index(route_id)]

    if model in ("mlp", "gru"):
        net, ck = _red(model)
        d = ventanas(Ys[[r]], cal, np.array([t0]), ck["hp"]["window"], H, solo_con_objetivo=False)
        with torch.no_grad():
            p, q = net(torch.from_numpy(d["past"]), torch.from_numpy(d["fut"]), torch.tensor([r]))
        point, qs = p.numpy()[0], q.numpy()[0]
    else:
        art = joblib.load(FORECASTING_ARTIFACTS / "baselines.joblib")[model]
        if model == "naive":
            point = B.naive_estacional(Ys, np.array([r]), np.array([t0]), H)[0]
        else:
            d = ventanas(Ys[[r]], cal, np.array([t0]), art["hp"]["window"], H, solo_con_objetivo=False)
            d["route"] = np.array([r])
            point = np.clip(art["model"].predict(B.matriz(d, len(meta["rutas"])))[0], 0, None)
        qs = B.con_cuantiles(point[None], art["offsets"])[0]

    upper = np.array([np.interp(quantile, levels, qs[h]) for h in range(H)])
    obs = Ys[r, t0 + 1:t0 + H + 1] * scale
    return pd.DataFrame({
        "date": fechas[t0 + 1:t0 + H + 1], "route_id": route_id, "point": point * scale,
        "upper_quantile": np.maximum(upper * scale, point * scale), "origin_date": origin,
        "model_version": meta["model_version"], "horizon": np.arange(1, H + 1), "quantile": quantile,
        "model": model, "observed": obs,
    }).head(horizon)
