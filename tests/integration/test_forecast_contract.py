"""Contrato forecast() para P2/P5: requiere artefactos entrenados (artifacts/forecasting)."""

import pandas as pd
import pytest

from transport_ai.common.paths import FORECASTING_ARTIFACTS

pytestmark = pytest.mark.skipif(not (FORECASTING_ARTIFACTS / "metadata.json").exists(), reason="sin artefactos M1")


@pytest.mark.parametrize("model", ["mlp", "gru", "ridge", "naive"])
def test_forecast_contrato(model):
    from transport_ai.forecasting.infer import forecast
    f = forecast("50001-11001", "2023-12-31", model=model)
    assert list(f.columns[:6]) == ["date", "route_id", "point", "upper_quantile", "origin_date", "model_version"]
    assert len(f) == 30
    assert (f["date"] == pd.date_range("2024-01-01", periods=30)).all()   # fechas posteriores al corte, en orden
    assert (f["point"] >= 0).all() and (f["upper_quantile"] >= f["point"]).all()
    assert f["observed"].isna().all()                                      # futuro desconocido


def test_cuantil_mayor_no_reduce_limite_y_horizonte_corto():
    from transport_ai.forecasting.infer import forecast
    a = forecast("11001-50001", "2023-11-30", quantile=0.75)
    b = forecast("11001-50001", "2023-11-30", quantile=0.95)
    assert (b["upper_quantile"] >= a["upper_quantile"] - 1e-6).all()
    assert len(forecast("11001-50001", "2023-11-30", horizon=7)) == 7


@pytest.mark.parametrize("kw", [{"route_id": "00000-00000"}, {"origin_date": "2024-02-01"},
                                {"quantile": 0.5}, {"horizon": 31}])
def test_forecast_rechaza_entradas_invalidas(kw):
    from transport_ai.forecasting.infer import forecast
    args = {"route_id": "50001-11001", "origin_date": "2023-12-31", **kw}
    with pytest.raises(ValueError):
        forecast(**args)
