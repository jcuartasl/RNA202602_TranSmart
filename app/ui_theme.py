from __future__ import annotations

import streamlit as st


def apply_transmart_theme() -> None:
    """Apply clean corporate TranSmart styling across all Streamlit pages."""
    st.markdown(
        """
        <style>
        .transmart-hero {
            background: linear-gradient(135deg, #0F2942 0%, #164E63 100%);
            color: #F8FAFC;
            padding: 1.5rem 1.75rem;
            border-radius: 12px;
            margin-bottom: 1.25rem;
            border-left: 6px solid #14B8A6;
        }
        .transmart-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.85rem;
            color: #FFFFFF;
        }
        .transmart-hero p {
            margin: 0;
            color: #CBD5E1;
            font-size: 0.98rem;
        }
        .transmart-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }
        .transmart-card h3 {
            margin-top: 0;
            color: #0F2942;
            font-size: 1.1rem;
        }
        .transmart-badge {
            display: inline-block;
            background-color: #CCFBF1;
            color: #0F766E;
            font-weight: 600;
            font-size: 0.78rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mock_banner(module_label: str) -> None:
    """Display a clear notice that the page is currently powered by mock contracts."""
    st.info(
        f"**Modo demostración (Mocks activos · {module_label}):** "
        "Los resultados mostrados son datos sintéticos generados mediante los contratos "
        "simulados en `transport_ai/common/mocks.py` mientras se integran los modelos entrenados.",
        icon="ℹ️",
    )


def render_sidebar_footer() -> None:
    """Render consistent sidebar status across pages."""
    with st.sidebar:
        st.markdown("### 🚌 TranSmart AI")
        st.caption("Plataforma integrada de analítica y redes neuronales.")
        st.divider()
        st.markdown("**Estado del entorno**")
        st.success("Contratos Mock activos", icon="✅")
        st.caption("Sin entrenamiento en vivo · Solo inferencia")
