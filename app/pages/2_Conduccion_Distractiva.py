from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.ui_theme import apply_transmart_theme, render_mock_banner, render_sidebar_footer
from transport_ai.common.mocks import (
    DRIVER_CLASS_LABELS,
    MAX_IMAGE_BYTES,
    classify_driver,
    generate_sample_driver_image,
)

st.set_page_config(
    page_title="M2 · Conducción Distractiva | TranSmart AI",
    page_icon="📷",
    layout="wide",
)

apply_transmart_theme()
render_sidebar_footer()

st.title("📷 Módulo 2: Detección de Conducción Distractiva")
render_mock_banner("Módulo 2 · Visión en Cabina")

st.warning(
    "**Uso experimental y privacidad:** Esta herramienta es de apoyo analítico y **no sustituye "
    "supervisión humana ni normativa vial**. Las imágenes cargadas se procesan en memoria y **no se "
    "almacenan en disco**.",
    icon="⚠️",
)

left_ctrl, right_ctrl = st.columns([1.4, 1.0])

with left_ctrl:
    uploaded_file = st.file_uploader(
        "Subir imagen de cabina (JPG/PNG · máx. 5 MB)",
        type=["jpg", "jpeg", "png"],
    )
    use_demo_img = st.checkbox(
        "Usar imagen sintética de ejemplo si no he subido una",
        value=True,
    )

with right_ctrl:
    alert_threshold = st.slider(
        "Umbral de alerta (probabilidad de distracción)",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Ajusta el equilibrio entre falsas alarmas y distracciones no detectadas.",
    )

image_bytes: bytes | None = None
if uploaded_file is not None:
    if uploaded_file.size > MAX_IMAGE_BYTES:
        st.error("El archivo supera el límite de 5 MB. Por favor sube una imagen más liviana.")
    else:
        image_bytes = uploaded_file.getvalue()
elif use_demo_img:
    image_bytes = generate_sample_driver_image()

if image_bytes is None:
    st.info("Sube una imagen JPG o PNG (o activa la imagen de ejemplo) para ejecutar el análisis.")
else:
    try:
        result = classify_driver(image_bytes=image_bytes, alert_threshold=alert_threshold)
    except ValueError as err:
        st.error(f"No se pudo procesar la imagen: {err}")
    else:
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Clase predicha", result["class_label"])
        kpi2.metric("Probabilidad de alerta", f"{result['alert_probability']:.1%}")
        kpi3.metric(
            "Decisión operativa",
            "🚨 ALERTA ACTIVA" if result["alert_bool"] else "✅ Sin alerta",
            delta=f"Umbral {alert_threshold:.0%}",
        )

        img_col1, img_col2 = st.columns(2, gap="medium")
        with img_col1:
            st.image(image_bytes, caption="Imagen analizada (en memoria)", use_container_width=True)
        with img_col2:
            st.image(
                result["gradcam_png_bytes"],
                caption="Mapa de atención (Grad-CAM simulado)",
                use_container_width=True,
            )

        st.subheader("Distribución de probabilidades por clase")
        probs_df = pd.DataFrame(
            [
                {
                    "Clase": DRIVER_CLASS_LABELS.get(cls, cls),
                    "Probabilidad": prob,
                }
                for cls, prob in result["class_probabilities"].items()
            ]
        ).set_index("Clase")
        st.bar_chart(probs_df, use_container_width=True)
