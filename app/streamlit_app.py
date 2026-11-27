from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.ui_theme import apply_transmart_theme, render_sidebar_footer
from transport_ai.common.mocks import MODEL_METADATA

st.set_page_config(
    page_title="TranSmart AI · Plataforma de Movilidad",
    page_icon="🚌",
    layout="wide",
)

apply_transmart_theme()
render_sidebar_footer()

st.markdown(
    """
    <div class="transmart-hero">
        <h1>🚌 TranSmart AI — Plataforma Integrada de Analítica y Redes Neuronales</h1>
        <p>
            Sistema multipágina de apoyo a la decisión para transporte intermunicipal:
            pronóstico de demanda, seguridad en cabina y recomendación de destinos turísticos.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "**Estado actual de la plataforma:** Módulo 3 (Recomendación turística · P4) conectado al "
    "modelo neuronal entrenado (`recommender_entities_v2`). Módulos 1 y 2 operan con contratos "
    "simulados (*mocks*) mientras P1–P3 entregan sus modelos.",
    icon="ℹ️",
)

st.subheader("Casos de negocio integrados")
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        """
        <div class="transmart-card">
            <span class="transmart-badge">Módulo 1 · Colombia (Mock)</span>
            <h3>📈 Demanda y Operación</h3>
            <p>
                Pronóstico de pasajeros a 30 días (puntual y cuantil P90) por corredor
                DIVIPOLA y conversión automática en despachos, flota y conductores diarios.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="transmart-card">
            <span class="transmart-badge">Módulo 2 · Visión (Mock)</span>
            <h3>📷 Conducción Distractiva</h3>
            <p>
                Clasificación de 5 comportamientos en cabina con umbral de alerta
                configurable y mapa de atención Grad-CAM para auditoría visual.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="transmart-card">
            <span class="transmart-badge">Módulo 3 · India (Integrado P4)</span>
            <h3>🧭 Recomendación Turística</h3>
            <p>
                Motor neuronal Top-K sobre catálogo por entidad (<code>place_0001</code>–<code>place_0005</code>)
                para viajeros con historial y <em>cold-start</em>, separado de la demanda colombiana.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

left, right = st.columns([1.4, 1.0], gap="large")

m3_artifact = ROOT_DIR / "artifacts/recommender_entities_v2/neural/best.pt"
metadata = {
    **MODEL_METADATA,
    "m3_recommender": {
        "version": "entities-v2-neural" if m3_artifact.exists() else MODEL_METADATA["m3_recommender"]["version"],
        "trained_at": (
            "Integrado (artifacts/recommender_entities_v2/neural/best.pt)"
            if m3_artifact.exists()
            else MODEL_METADATA["m3_recommender"]["trained_at"]
        ),
        "owner": "P4",
    },
}

with left:
    st.subheader("Acerca de y Trazabilidad de Modelos")
    meta_rows = [
        {
            "Módulo": key,
            "Versión": val["version"],
            "Estado / Artefacto": val["trained_at"],
            "Responsable": val["owner"],
        }
        for key, val in metadata.items()
    ]
    st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("Documentación y Entregables")
    st.markdown(
        """
        - 📄 **Reporte técnico / Blog:** `reports/blog.md` · `reports/blog_p4_borrador.md`
        - 📚 **Referencias (APA 7 / BibTeX):** `reports/citations.bib`
        - 🛡️ **Ética e IA Responsable:** `docs/ethics_ai.md`
        - 🎥 **Videos y guion de demo:** `videos/README.md` · `docs/demo_script.md`
        - 🏗️ **Arquitectura y contratos:** `docs/architecture.md` · `transport_ai/recommender/ENTREGA_P5.md`
        """
    )
