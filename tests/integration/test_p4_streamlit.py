from __future__ import annotations

from pathlib import Path
from streamlit.testing.v1 import AppTest

from transport_ai.recommender.infer import recommend

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data/processed/recommender_entities_v2"
NEURAL_ARTIFACT = REPO_ROOT / "artifacts/recommender_entities_v2/neural/best.pt"


def test_p4_neural_recommender_cold_start_and_existing_user() -> None:
    cold = recommend(
        user_id=None,
        profile={"preferences": "Beach, Historical"},
        k=3,
        method="neural",
        processed=PROCESSED_DIR,
        neural_artifact=NEURAL_ARTIFACT,
    )
    assert len(cold) == 3
    assert all(item["history"] == [] for item in cold)
    assert len({item["destination_id"] for item in cold}) == 3

    existing = recommend(
        user_id="1",
        profile=None,
        k=3,
        method="neural",
        processed=PROCESSED_DIR,
        neural_artifact=NEURAL_ARTIFACT,
    )
    assert len(existing) == 3
    assert len({item["destination_id"] for item in existing}) == 3


def test_streamlit_page3_uses_real_p4_recommender() -> None:
    at = AppTest.from_file(str(REPO_ROOT / "app/pages/3_Recomendacion_Destinos.py"))
    at.run(timeout=20)
    assert not at.exception
    assert len(at.dataframe) >= 1
