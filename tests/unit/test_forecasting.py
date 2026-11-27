"""Tests M1 (P1): calendario, faltantes, fuga temporal, referencias, redes, pérdida y métricas."""

import numpy as np
import pandas as pd
import pytest
import torch

from transport_ai.forecasting import baselines as B
from transport_ai.forecasting import evaluation as E
from transport_ai.forecasting.cli import aplicar_set
from transport_ai.forecasting.data import serie_completa
from transport_ai.forecasting.features import CAL_COLS, calendario, festivos, origenes, ventanas
from transport_ai.forecasting.models import crear, perdida

H, W = 30, 14


def test_festivos_colombia_2023_ley_emiliani():
    esperados = ["2023-01-01", "2023-01-09", "2023-03-20", "2023-04-06", "2023-04-07", "2023-05-01",
                 "2023-05-22", "2023-06-12", "2023-06-19", "2023-07-03", "2023-07-20", "2023-08-07",
                 "2023-08-21", "2023-10-16", "2023-11-06", "2023-11-13", "2023-12-08", "2023-12-25"]
    assert festivos(2023) == set(pd.to_datetime(esperados))


def test_calendario_vispera_de_puente():
    c = calendario(pd.date_range("2023-01-01", "2023-01-31"))
    assert c.loc["2023-01-06", "vispera"] == 1  # viernes antes del lunes festivo de Reyes
    assert c.loc["2023-01-09", "festivo"] == 1 and c.loc["2023-01-10", "festivo"] == 0
    assert list(c.columns) == CAL_COLS


def test_serie_distingue_cero_real_de_dia_desconocido():
    d = pd.DataFrame({
        "fecha": pd.to_datetime(["2023-01-01", "2023-01-03", "2023-01-04"]),
        "origen": "11001", "destino": "50001", "tipo": "ORIGEN",
        "pasajeros": [10, 0, 0], "despachos": [1, 2, 0], "registros": 1, "reg_pax0": [0, 1, 0],
        "reg_exceso": 0, "pax_exceso": 0,
    })
    s = serie_completa(d, "2023-01-01", "2023-01-04").set_index("fecha")
    assert len(s) == 4                                   # calendario continuo
    assert np.isnan(s.loc["2023-01-02", "pasajeros"])    # sin registro -> desconocido, no cero
    assert s.loc["2023-01-02", "anomalia"] == "sin_registro"
    assert np.isnan(s.loc["2023-01-03", "pasajeros"])    # 0 pasajeros con 2 despachos -> desconocido
    assert s.loc["2023-01-04", "pasajeros"] == 0         # cero sin despachos -> cero real
    assert s["valido"].tolist() == [True, False, False, True]


def _panel(T=200, R=3, seed=0):
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2022-01-01", periods=T)
    return rng.poisson(100, (R, T)).astype(float), fechas, calendario(fechas).to_numpy()


def test_ventana_nunca_usa_futuro():
    Y, _, cal = _panel()
    t0 = np.array([60, 100])
    a = ventanas(Y, cal, t0, W, H)
    Y2 = Y.copy()
    Y2[:, 61:] = -999                                    # alterar todo lo posterior al primer origen
    b = ventanas(Y2, cal, t0, W, H)
    assert np.array_equal(a["past"][a["t0"] == 60], b["past"][b["t0"] == 60])
    assert np.array_equal(a["fut"], b["fut"])            # calendario futuro no depende de la demanda
    assert np.array_equal(a["y"][0], Y[0, 61:61 + H])    # objetivo = días t0+1..t0+H


def test_origenes_no_cruzan_particiones():
    _, fechas, _ = _panel(T=300)
    tr = origenes(fechas, "2022-01-01", "2022-06-30", W, H)
    va = origenes(fechas, "2022-07-01", "2022-09-30", W, H)
    assert fechas[tr + H].max() <= pd.Timestamp("2022-06-30")
    assert fechas[va + 1].min() >= pd.Timestamp("2022-07-01")
    assert (tr - W + 1).min() >= 0


def test_naive_estacional_repite_ultima_semana():
    Y = np.arange(100, dtype=float)[None]
    p = B.naive_estacional(Y, np.array([0]), np.array([50]), H)[0]
    assert p[0] == 44 and p[6] == 50 and p[7] == 44      # h=1 -> t0-6, h=8 -> t0-6


def test_naive_estacional_salta_faltantes():
    Y = np.arange(100, dtype=float)[None]
    Y[0, 44] = np.nan
    assert B.naive_estacional(Y, np.array([0]), np.array([50]), H)[0][0] == 37


def test_cuantiles_baseline_no_cruzan():
    rng = np.random.default_rng(0)
    point = rng.random((50, H))
    off = B.offsets_cuantiles(point + rng.normal(0, 1, (50, H)), point, [0.75, 0.9, 0.95])
    q = B.con_cuantiles(point, off)
    assert (q >= point[..., None]).all() and (np.diff(q, axis=-1) >= 0).all()


@pytest.mark.parametrize("arch,kw", [("mlp", {"hidden": [16]}), ("gru", {"hidden": 8})])
def test_redes_salida_no_negativa_y_cuantiles_monotonos(arch, kw):
    F = len(CAL_COLS)
    m = crear(arch, window=W, n_feat=2 + F, n_cal=F, horizon=H, n_routes=4, n_q=5, **kw)
    p, q = m(torch.randn(8, W, 2 + F), torch.randn(8, H, F), torch.randint(0, 4, (8,)))
    assert p.shape == (8, H) and q.shape == (8, H, 5)
    assert (p >= 0).all() and (q >= p[..., None]).all() and (q.diff(dim=-1) >= 0).all()


@pytest.mark.parametrize("kind", ["mae", "poisson"])
def test_perdida_ignora_dias_desconocidos(kind):
    lv = torch.tensor([0.75, 0.9])
    p, q = torch.rand(4, H) + 0.1, torch.rand(4, H, 2) + 1
    y = torch.rand(4, H)
    y[0, :5] = float("nan")
    base = perdida(p, q, y, lv, kind)
    p2, q2 = p.clone(), q.clone()
    p2[0, :5] += 10                                      # cambiar la predicción solo en días desconocidos
    q2[0, :5] += 10
    assert torch.isfinite(base) and torch.allclose(base, perdida(p2, q2, y, lv, kind))


def test_mase_serie_constante_no_definido():
    Y = np.ones((2, 60))
    Y[1] = np.arange(60) % 5                             # período 5: no se cancela con m=7
    d = E.denominador_mase(Y, pd.date_range("2022-01-01", periods=60), "2022-01-01", "2022-03-01")
    assert np.isnan(d[0]) and d[1] > 0
    df = pd.DataFrame({"route_id": ["a", "b"], "y": [1.0, 2.0], "point": [1.0, 1.0], "p90": [2.0, 2.0]})
    m = E.metricas(df, {"a": np.nan, "b": 1.0})
    assert m["mase"] == 1.0 and m["cobertura90"] == 1.0


def test_set_override_valida_claves():
    cfg = {"train": {"lr": 0.001}}
    assert aplicar_set(cfg, ["train.lr=0.003"])["train"]["lr"] == 0.003
    with pytest.raises(KeyError):
        aplicar_set(cfg, ["train.lrr=1"])
