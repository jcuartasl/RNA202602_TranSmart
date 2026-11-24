from __future__ import annotations

from datetime import date
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from transport_ai.common.mocks import (
    MAX_IMAGE_BYTES,
    classify_driver,
    forecast,
    generate_sample_driver_image,
    historical_demand,
    plan_operations,
    recommend,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_forecast_and_operations_contract() -> None:
    origin = date(2026, 11, 1)
    hist_df = historical_demand("BOG-MED", origin_date=origin, days=30)
    assert len(hist_df) == 30
    assert (hist_df["observed_passengers"] >= 0).all()

    fc_df = forecast("BOG-MED", origin_date=origin, horizon=30, quantile=0.90)
    assert len(fc_df) == 30
    assert (fc_df["point_forecast"] >= 0).all()
    assert (fc_df["quantile_forecast"] >= fc_df["point_forecast"]).all()

    ops_high_occ = plan_operations(fc_df, occupancy_target=0.85)
    ops_low_occ = plan_operations(fc_df, occupancy_target=0.60)
    assert ops_low_occ["avg_daily_dispatches"] >= ops_high_occ["avg_daily_dispatches"]
    assert ops_high_occ["peak_day_vehicles"] >= 1
    assert ops_high_occ["peak_day_drivers"] >= ops_high_occ["peak_day_vehicles"]


def test_classify_driver_contract_and_guards() -> None:
    sample_png = generate_sample_driver_image()
    res_low = classify_driver(sample_png, alert_threshold=0.05)
    res_high = classify_driver(sample_png, alert_threshold=0.99)

    assert len(res_low["class_probabilities"]) == 5
    assert pytest.approx(sum(res_low["class_probabilities"].values()), abs=1e-3) == 1.0
    assert res_low["alert_bool"] is True
    assert res_high["alert_bool"] is False
    assert res_low["gradcam_png_bytes"].startswith(b"\x89PNG")

    with pytest.raises(ValueError):
        classify_driver(b"")
    with pytest.raises(ValueError):
        classify_driver(b"x" * (MAX_IMAGE_BYTES + 1))
    with pytest.raises(ValueError):
        classify_driver(b"not-an-image")


def test_recommend_cold_start_and_preferences() -> None:
    cold = recommend(user_id=None, profile={"preferences": ["playa"]}, k=4)
    assert cold["is_cold_start"] is True
    assert cold["history"] == []
    assert len(cold["recommendations"]) == 4
    assert cold["recommendations"][0]["category"] == "playa"

    existing = recommend(user_id="U-101", profile={"preferences": ["aventura"]}, k=3)
    assert existing["is_cold_start"] is False
    assert len(existing["history"]) > 0
    assert len(existing["recommendations"]) == 3


@pytest.mark.parametrize(
    "app_rel_path",
    [
        "app/streamlit_app.py",
        "app/pages/1_Demanda_y_Operacion.py",
        "app/pages/2_Conduccion_Distractiva.py",
        "app/pages/3_Recomendacion_Destinos.py",
    ],
)
def test_streamlit_pages_render_without_exception(app_rel_path: str) -> None:
    at = AppTest.from_file(str(REPO_ROOT / app_rel_path))
    at.run(timeout=15)
    assert not at.exception
