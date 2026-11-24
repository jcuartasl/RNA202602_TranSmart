from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.ui_theme import apply_transmart_theme, render_mock_banner, render_sidebar_footer
from transport_ai.common.mocks import USER_HISTORIES, recommend

st.set_page_config(
    page_title="M3 · Recomendación de Destinos | TranSmart AI",
    page_icon="🧭",
    layout="wide",
)

apply_transmart_theme()
render_sidebar_footer()

st.title("🧭 Módulo 3: Recomendación de Destinos Turísticos (India)")
render_mock_banner("Módulo 3 · Recomendador Turístico")

ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.5, 1.0])

with ctrl1:
    mode = st.radio(
        "Tipo de viajero",
        options=["Usuario existente", "Nuevo perfil (cold-start)"],
    )
    user_id = (
        st.selectbox("Seleccionar ID de usuario", options=list(USER_HISTORIES.keys()))
        if mode == "Usuario existente"
        else None
    )

with ctrl2:
    prefs = st.multiselect(
        "Preferencias del viajero (afectan el ranking)",
        options=["cultura", "playa", "naturaleza", "aventura", "gastronomia"],
        default=["cultura"],
    )

with ctrl3:
    k = st.slider("Cantidad de recomendaciones (K)", min_value=1, max_value=8, value=5)

result = recommend(user_id=user_id, profile={"preferences": prefs}, k=k)

if result["is_cold_start"]:
    st.info(
        "**Usuario sin interacciones previas (cold-start):** "
        "el ranking se genera combinando preferencias declaradas y popularidad base sin simular historial.",
        icon="🆕",
    )
else:
    st.success(
        f"**Historial observado ({result['user_id']}):** " + " · ".join(result["history"]),
        icon="🧳",
    )

st.subheader(f"Top {k} destinos recomendados")
rec_df = pd.DataFrame(result["recommendations"]).rename(
    columns={
        "destination_id": "ID",
        "destination_name": "Destino",
        "category": "Categoría",
        "region": "Región",
        "score": "Puntaje",
        "reason": "Justificación",
    }
)
st.dataframe(rec_df, use_container_width=True, hide_index=True)
