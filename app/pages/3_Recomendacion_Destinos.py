from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.ui_theme import apply_transmart_theme, render_sidebar_footer
from transport_ai.recommender.infer import recommend

PROCESSED_DIR = ROOT_DIR / "data/processed/recommender_entities_v2"
NEURAL_ARTIFACT = ROOT_DIR / "artifacts/recommender_entities_v2/neural/best.pt"
KNN_ARTIFACT = ROOT_DIR / "artifacts/recommender_entities_v2/item_knn.json"

PREFERENCE_OPTIONS = ["Beach", "Historical", "Nature", "Adventure", "City"]

METHOD_LABELS = {
    "neural": "Red Neuronal Dos Torres (P4 · recomendado)",
    "neural-hybrid": "Neuronal + bono de preferencias (experimental)",
    "item-knn": "Baseline Item-KNN (requiere historial)",
    "content": "Baseline Contenido (preferencias)",
    "popularity": "Baseline Popularidad",
}

st.set_page_config(
    page_title="M3 · Recomendación de Destinos | TranSmart AI",
    page_icon="🧭",
    layout="wide",
)

apply_transmart_theme()
render_sidebar_footer()

st.title("🧭 Módulo 3: Recomendación de Destinos Turísticos (India)")
st.success(
    "**Modelo real integrado (P4 · `recommender_entities_v2`):** "
    "inferencia sobre el catálogo por entidad lógica (`place_0001`–`place_0005`), "
    "separada del dominio de demanda colombiana.",
    icon="✅",
)


@st.cache_data
def _load_recommender_metadata(processed_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(processed_dir)
    catalog_df = pd.read_csv(base / "catalog.csv")
    profiles_df = pd.read_csv(base / "profiles.csv")
    profiles_df["user_id"] = profiles_df["user_id"].astype(str)
    return catalog_df, profiles_df


required_files = [
    PROCESSED_DIR / "catalog.csv",
    PROCESSED_DIR / "profiles.csv",
    PROCESSED_DIR / "split_manifest.csv",
    NEURAL_ARTIFACT,
]
missing = [str(p.relative_to(ROOT_DIR)) for p in required_files if not p.exists()]

if missing:
    st.error(
        "Faltan artefactos requeridos para el recomendador de P4: "
        + ", ".join(f"`{m}`" for m in missing)
    )
    st.stop()

catalog_df, profiles_df = _load_recommender_metadata(str(PROCESSED_DIR))
catalog_by_id = catalog_df.set_index("destination_id").to_dict(orient="index")
user_ids = sorted(profiles_df["user_id"].tolist(), key=lambda x: (len(x), x))
profiles_by_id = profiles_df.set_index("user_id").to_dict(orient="index")

ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.5, 1.0])

with ctrl1:
    mode = st.radio(
        "Tipo de viajero",
        options=["Usuario existente", "Nuevo perfil (cold-start)"],
    )
    if mode == "Usuario existente":
        user_id: str | None = st.selectbox("Seleccionar ID de usuario", options=user_ids)
        stored_prefs = str(profiles_by_id[user_id].get("preferences", ""))
        st.caption(f"Preferencias registradas del usuario: `{stored_prefs}`")
    else:
        user_id = None

with ctrl2:
    default_prefs = ["Beach", "Historical"] if user_id is None else []
    selected_prefs = st.multiselect(
        "Preferencias del viajero (Beach, Historical, Nature, Adventure, City)",
        options=PREFERENCE_OPTIONS,
        default=default_prefs,
        help="Para usuario existente, si lo dejas vacío se usan sus preferencias registradas.",
    )
    available_methods = (
        ["neural", "neural-hybrid", "content", "popularity"]
        if user_id is None
        else list(METHOD_LABELS.keys())
    )
    method = st.selectbox(
        "Método de recomendación",
        options=available_methods,
        format_func=lambda m: METHOD_LABELS[m],
    )

with ctrl3:
    max_k = max(1, len(catalog_df))
    k = st.slider(
        "Cantidad de recomendaciones (K)",
        min_value=1,
        max_value=max_k,
        value=min(3, max_k),
    )

if user_id is None and not selected_prefs:
    st.warning("Selecciona al menos una preferencia para el nuevo perfil (cold-start).")
    st.stop()

profile_arg = {"preferences": ", ".join(selected_prefs)} if selected_prefs else None

try:
    recommendations = recommend(
        user_id=user_id,
        profile=profile_arg,
        k=k,
        method=method,
        processed=PROCESSED_DIR,
        artifact=KNN_ARTIFACT,
        neural_artifact=NEURAL_ARTIFACT,
    )
except Exception as exc:
    st.error(f"Error al ejecutar inferencia del recomendador: {exc}")
    st.stop()

history = recommendations[0]["history"] if recommendations else []

if user_id is None:
    st.info(
        "**Usuario sin interacciones previas (cold-start):** "
        "no se simula historial; el ranking usa únicamente las preferencias declaradas.",
        icon="🆕",
    )
elif history:
    st.success(
        f"**Historial observado en entrenamiento (`{user_id}`):** " + " · ".join(history),
        icon="🧳",
    )
else:
    st.info(
        f"**Usuario `{user_id}` sin interacciones positivas en entrenamiento:** "
        "se evalúa con su perfil registrado y preferencias.",
        icon="ℹ️",
    )

st.subheader(f"Top {len(recommendations)} destinos recomendados")

rows = []
for rank, item in enumerate(recommendations, start=1):
    meta = catalog_by_id.get(item["destination_id"], {})
    rows.append(
        {
            "Ranking": rank,
            "ID": item["destination_id"],
            "Destino": item["destination_name"],
            "Estado / Región": meta.get("state", "—"),
            "Categoría": meta.get("type", "—"),
            "Mejor época": meta.get("best_time_to_visit", "—"),
            "Puntaje (ordenamiento)": round(float(item["score"]), 4),
            "Justificación": item["reason"],
        }
    )

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("Catálogo lógico disponible y límites del modelo (P4)"):
    st.dataframe(catalog_df, use_container_width=True, hide_index=True)
    st.caption(
        "Nota: `Puntaje (ordenamiento)` es una salida del modelo para rankear destinos y "
        "**no** representa una probabilidad calibrada de compra. Datos de origen Kaggle sin "
        "verificación de procedencia real/sintética; no usar para planear flota colombiana."
    )
