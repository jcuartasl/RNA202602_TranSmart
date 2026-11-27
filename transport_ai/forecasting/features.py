"""Calendario colombiano y tensores de ventana para M1.

Todas las variables exógenas son de calendario: se conocen con años de anticipación, así que
usarlas para los 30 días futuros no es fuga.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

# Ley 51 de 1983 ("Ley Emiliani"): festivos fijos y festivos trasladables al lunes siguiente.
FIJOS = [(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)]
TRASLADABLES = [(1, 6), (3, 19), (6, 29), (8, 15), (10, 12), (11, 1), (11, 11)]
PASCUA = {-3: False, -2: False, 43: True, 64: True, 71: True}  # días desde Pascua: Jueves/Viernes Santo, Ascensión, Corpus, Sagrado Corazón (ya en lunes)
CAL_COLS = [f"dow_{i}" for i in range(7)] + ["festivo", "vispera", "semana_santa", "fin_ano",
                                             "vacaciones_mitad", "doy_sin", "doy_cos"]


def festivos(anio: int) -> set[pd.Timestamp]:
    out = {pd.Timestamp(anio, m, d) for m, d in FIJOS}
    for m, d in TRASLADABLES:
        f = pd.Timestamp(anio, m, d)
        out.add(f + pd.Timedelta(days=(7 - f.dayofweek) % 7))  # lunes siguiente (o el mismo lunes)
    e = pd.Timestamp(easter(anio))
    out |= {e + pd.Timedelta(days=k) for k in PASCUA}
    return out


def calendario(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables de calendario por día (índice = fecha, columnas = CAL_COLS)."""
    fest = set().union(*(festivos(a) for a in range(fechas.min().year - 1, fechas.max().year + 2)))
    es_fest = fechas.isin(list(fest))
    manana = fechas + pd.Timedelta(days=1)
    pascua = pd.to_datetime([easter(y) for y in fechas.year])
    md = fechas.month * 100 + fechas.day
    c = pd.DataFrame(index=fechas)
    for i in range(7):
        c[f"dow_{i}"] = (fechas.dayofweek == i)
    c["festivo"] = es_fest
    # víspera: mañana es festivo, o viernes antes de un lunes festivo (inicio de puente)
    c["vispera"] = manana.isin(list(fest)) | ((fechas.dayofweek == 4) & (fechas + pd.Timedelta(days=3)).isin(list(fest)))
    c["semana_santa"] = ((fechas - pascua).days >= -7) & ((fechas - pascua).days <= 0)  # Domingo de Ramos..Pascua
    c["fin_ano"] = (md >= 1215) | (md <= 112)
    c["vacaciones_mitad"] = (md >= 615) & (md <= 715)  # supuesto: calendario escolar A aproximado
    doy = 2 * np.pi * fechas.dayofyear / 365.25
    c["doy_sin"], c["doy_cos"] = np.sin(doy), np.cos(doy)
    return c.astype("float32")


def panel(series: pd.DataFrame, rutas: list[str]) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Matriz (rutas × días) de pasajeros con NaN en días desconocidos."""
    p = series.pivot(index="route_id", columns="fecha", values="pasajeros").loc[rutas]
    return p.to_numpy(dtype="float64"), pd.DatetimeIndex(p.columns)


def origenes(fechas: pd.DatetimeIndex, inicio: str, fin: str, window: int, horizon: int) -> np.ndarray:
    """Índices t0 (último día observado) cuyos 30 objetivos t0+1..t0+H caen completos en [inicio, fin]."""
    t = np.arange(len(fechas))
    ok = (t - window + 1 >= 0) & (t + horizon < len(fechas))
    ok[ok] &= (fechas[t[ok] + 1] >= pd.Timestamp(inicio)) & (fechas[t[ok] + horizon] <= pd.Timestamp(fin))
    return t[ok]


def ventanas(Y: np.ndarray, cal: np.ndarray, t0: np.ndarray, window: int, horizon: int,
             solo_con_objetivo: bool = True) -> dict:
    """Muestras (ruta, origen). Y ya escalada por ruta; cal = (días, F).

    past: (N, window, 2+F) = [y escalada (NaN→0), máscara observada, calendario] para t0-window+1..t0
    fut:  (N, horizon, F)  = calendario de t0+1..t0+horizon (conocido de antemano)
    y:    (N, horizon)     = objetivo escalado, NaN si desconocido (se enmascara en pérdida y métricas)
    """
    R = Y.shape[0]
    rr, tt = np.repeat(np.arange(R), len(t0)), np.tile(t0, R)
    lag = tt[:, None] + np.arange(-window + 1, 1)
    lead = tt[:, None] + np.arange(1, horizon + 1)
    yp = Y[rr[:, None], lag]
    past = np.concatenate([np.nan_to_num(yp)[..., None], (~np.isnan(yp))[..., None], cal[lag]], axis=-1)
    y = Y[rr[:, None], lead]
    keep = ~np.isnan(y).all(axis=1) if solo_con_objetivo else np.ones(len(y), bool)
    return {"past": past[keep].astype("float32"), "fut": cal[lead][keep].astype("float32"),
            "route": rr[keep], "t0": tt[keep], "y": y[keep].astype("float32")}
