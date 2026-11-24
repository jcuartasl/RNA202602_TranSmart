from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from app.ui_theme import apply_transmart_theme, render_mock_banner, render_sidebar_footer
from transport_ai.common.mocks import ROUTES, forecast, historical_demand, plan_operations

st.set_page_config(
    page_title="M1 · Demanda y Operación | TranSmart AI",
    page_icon="📈",
    layout="wide",
)

apply_transmart_theme()
render_sidebar_footer()

st.title("📈 Módulo 1: Pronóstico de Demanda y Operación (Colombia)")
render_mock_banner("Módulo 1 · Demanda y Operación")

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.8, 1.2, 1.2, 1.2])

with ctrl_col1:
    route_id = st.selectbox(
        "Corredor / Ruta (DIVIPOLA)",
        options=list(ROUTES.keys()),
        format_func=lambda r: f"{r} — {ROUTES[r]['name']}",
    )

with ctrl_col2:
    origin_date = st.date_input(
        "Fecha origen del pronóstico",
        value=date(2026, 11, 1),
    )

with ctrl_col3:
    occupancy_target = st.slider(
        "Ocupación objetivo",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05,
        help="Control operativo: ajusta la ocupación promedio esperada por despacho.",
    )

with ctrl_col4:
    quantile = st.select_slider(
        "Cuantil de planeación",
        options=[0.50, 0.75, 0.90, 0.95],
        value=0.90,
        help="Control operativo: P90 cubre picos de demanda con menor riesgo de desabastecimiento.",
    )

recalc_clicked = st.button("🔄 Recalcular solo motor de operación", type="primary")
if recalc_clicked:
    st.toast("Operación recalculada con los parámetros actuales.", icon="✅")

hist_df = historical_demand(route_id=route_id, origin_date=origin_date, days=30)
fc_df = forecast(route_id=route_id, origin_date=origin_date, horizon=30, quantile=quantile)
ops = plan_operations(fc_df, occupancy_target=occupancy_target, bus_capacity=40)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Pasajeros 30d (Puntual)",
    f"{ops['total_passengers_point']:,}",
)
kpi2.metric(
    f"Pasajeros 30d (P{int(quantile * 100)})",
    f"{ops['total_passengers_quantile']:,}",
)
kpi3.metric(
    "Despachos / día (prom.)",
    f"{ops['avg_daily_dispatches']}",
)
kpi4.metric(
    "Pico vehículos / conductores",
    f"{ops['peak_day_vehicles']} veh · {ops['peak_day_drivers']} cond",
)

st.subheader("Serie histórica (30 días) y pronóstico a 30 días")

hist_plot = hist_df.set_index("date")[["observed_passengers"]].rename(
    columns={"observed_passengers": "Histórico observado"}
)
fc_plot = fc_df.set_index("date")[["point_forecast", "quantile_forecast"]].rename(
    columns={
        "point_forecast": "Pronóstico puntual",
        "quantile_forecast": f"Pronóstico P{int(quantile * 100)}",
    }
)
combined_chart = pd.concat([hist_plot, fc_plot], axis=0)
st.line_chart(combined_chart, use_container_width=True)

st.subheader("Plan operativo diario (30 días)")
schedule_display = ops["schedule"].rename(
    columns={
        "date": "Fecha",
        "route_id": "Ruta",
        "point_forecast": "Pasajeros (Puntual)",
        "quantile_forecast": f"Pasajeros (P{int(quantile * 100)})",
        "dispatches": "Despachos",
        "vehicles_required": "Vehículos requeridos",
        "drivers_required": "Conductores requeridos",
    }
)
st.dataframe(schedule_display, use_container_width=True, hide_index=True)
