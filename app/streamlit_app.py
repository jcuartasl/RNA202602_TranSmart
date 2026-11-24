from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.ui_theme import apply_transmart_theme, render_mock_banner, render_sidebar_footer
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

render_mock_banner("Portada")

st.subheader("Casos de negocio integrados")
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        """
        <div class="transmart-card">
            <span class="transmart-badge">Módulo 1 · Colombia</span>
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
            <span class="transmart-badge">Módulo 2 · Visión</span>
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
            <span class="transmart-badge">Módulo 3 · India (Separado)</span>
            <h3>🧭 Recomendación Turística</h3>
            <p>
                Motor Top-K de destinos turísticos para viajeros con historial y usuarios
                nuevos (<em>cold-start</em>), sin mezclar dominios con la demanda colombiana.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

left, right = st.columns([1.4, 1.0], gap="large")

with left:
    st.subheader("Acerca de y Trazabilidad de Modelos")
    meta_rows = [
        {
            "Módulo": key,
            "Versión": val["version"],
            "Fecha / Estado de entrenamiento": val["trained_at"],
            "Responsable": val["owner"],
        }
        for key, val in MODEL_METADATA.items()
    ]
    st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

with right:
    st.subheader("Documentación y Entregables")
    st.markdown(
        """
        - 📄 **Reporte técnico / Blog:** `reports/blog.md`
        - 📚 **Referencias (APA 7 / BibTeX):** `reports/citations.bib`
        - 🛡️ **Ética e IA Responsable:** `docs/ethics_ai.md`
        - 🎥 **Videos y guion de demo:** `videos/README.md` · `docs/demo_script.md`
        - 🏗️ **Arquitectura y contratos:** `docs/architecture.md`
        """
    )
