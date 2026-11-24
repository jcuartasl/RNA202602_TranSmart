from __future__ import annotations

from datetime import date, timedelta
import hashlib
import io
import math
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB privacy & memory limit

ROUTES: dict[str, dict[str, Any]] = {
    "BOG-MED": {
        "name": "Bogotá (11001) → Medellín (05001)",
        "origin_divipola": "11001",
        "dest_divipola": "05001",
        "base_demand": 1450,
        "distance_km": 415,
    },
    "BOG-CAL": {
        "name": "Bogotá (11001) → Cali (76001)",
        "origin_divipola": "11001",
        "dest_divipola": "76001",
        "base_demand": 1180,
        "distance_km": 462,
    },
    "MED-CTG": {
        "name": "Medellín (05001) → Cartagena (13001)",
        "origin_divipola": "05001",
        "dest_divipola": "13001",
        "base_demand": 890,
        "distance_km": 635,
    },
    "CAL-BAQ": {
        "name": "Cali (76001) → Barranquilla (08001)",
        "origin_divipola": "76001",
        "dest_divipola": "08001",
        "base_demand": 640,
        "distance_km": 985,
    },
}

DRIVER_CLASSES: list[str] = [
    "conduccion_segura",
    "giro",
    "escribir_telefono",
    "hablar_telefono",
    "otro_riesgo",
]

DRIVER_CLASS_LABELS: dict[str, str] = {
    "conduccion_segura": "Conducción segura",
    "giro": "Giro / maniobra",
    "escribir_telefono": "Escribir en teléfono",
    "hablar_telefono": "Hablar por teléfono",
    "otro_riesgo": "Otro riesgo / distracción",
}

ALERT_CLASSES: set[str] = {"escribir_telefono", "hablar_telefono", "otro_riesgo"}

DESTINATIONS: list[dict[str, Any]] = [
    {
        "destination_id": "IND-01",
        "destination_name": "Jaipur (Palacios y Fuerte Amber)",
        "category": "cultura",
        "region": "Rajastán",
        "base_popularity": 0.92,
    },
    {
        "destination_id": "IND-02",
        "destination_name": "Goa (Playas y descanso costero)",
        "category": "playa",
        "region": "Costa Oeste",
        "base_popularity": 0.90,
    },
    {
        "destination_id": "IND-03",
        "destination_name": "Rishikesh (Naturaleza y bienestar)",
        "category": "naturaleza",
        "region": "Uttarakhand",
        "base_popularity": 0.85,
    },
    {
        "destination_id": "IND-04",
        "destination_name": "Agra (Taj Mahal y patrimonio)",
        "category": "cultura",
        "region": "Uttar Pradesh",
        "base_popularity": 0.95,
    },
    {
        "destination_id": "IND-05",
        "destination_name": "Kerala (Backwaters y ecoturismo)",
        "category": "naturaleza",
        "region": "Sur de India",
        "base_popularity": 0.88,
    },
    {
        "destination_id": "IND-06",
        "destination_name": "Udaipur (Lagos y arquitectura)",
        "category": "cultura",
        "region": "Rajastán",
        "base_popularity": 0.83,
    },
    {
        "destination_id": "IND-07",
        "destination_name": "Manali (Montaña y aventura)",
        "category": "aventura",
        "region": "Himachal Pradesh",
        "base_popularity": 0.86,
    },
    {
        "destination_id": "IND-08",
        "destination_name": "Amritsar (Templo Dorado y gastronomía)",
        "category": "gastronomia",
        "region": "Punyak / Punjab",
        "base_popularity": 0.81,
    },
]

USER_HISTORIES: dict[str, list[str]] = {
    "U-101": ["Agra (Taj Mahal y patrimonio)", "Jaipur (Palacios y Fuerte Amber)"],
    "U-204": ["Goa (Playas y descanso costero)", "Kerala (Backwaters y ecoturismo)"],
    "U-309": ["Manali (Montaña y aventura)", "Rishikesh (Naturaleza y bienestar)"],
}

MODEL_METADATA: dict[str, dict[str, str]] = {
    "m1_forecast": {
        "version": "0.1.0-mock",
        "trained_at": "Simulación (sin artefacto entrenado)",
        "owner": "P1 / P2",
    },
    "m2_vision": {
        "version": "0.1.0-mock",
        "trained_at": "Simulación (sin artefacto entrenado)",
        "owner": "P3",
    },
    "m3_recommender": {
        "version": "0.1.0-mock",
        "trained_at": "Simulación (sin artefacto entrenado)",
        "owner": "P4",
    },
}


def _seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def historical_demand(
    route_id: str,
    origin_date: date,
    days: int = 30,
) -> pd.DataFrame:
    """Return 30-day synthetic historical passenger series ending on origin_date."""
    if route_id not in ROUTES:
        raise ValueError(f"Ruta desconocida: {route_id}")

    base = ROUTES[route_id]["base_demand"]
    rng = np.random.default_rng(_seed_from_text(f"hist:{route_id}:{origin_date.isoformat()}"))
    rows = []
    for offset in range(-days + 1, 1):
        d = origin_date + timedelta(days=offset)
        weekend_boost = 1.18 if d.weekday() >= 4 else 0.94
        seasonal = 1.0 + 0.08 * math.sin(2 * math.pi * d.timetuple().tm_yday / 30.0)
        noise = float(rng.normal(0.0, base * 0.04))
        demand = max(0, int(round(base * weekend_boost * seasonal + noise)))
        rows.append({"date": d, "route_id": route_id, "observed_passengers": demand, "is_mock": True})
    return pd.DataFrame(rows)


def forecast(
    route_id: str,
    origin_date: date,
    horizon: int = 30,
    quantile: float = 0.90,
) -> pd.DataFrame:
    """Mock contract for P1: returns `horizon` non-negative daily predictions with quantile >= point."""
    if route_id not in ROUTES:
        raise ValueError(f"Ruta desconocida: {route_id}")
    if horizon < 1:
        raise ValueError("El horizonte debe ser >= 1.")
    if not (0.50 <= quantile <= 0.99):
        raise ValueError("El cuantil debe estar entre 0.50 y 0.99.")

    base = ROUTES[route_id]["base_demand"]
    rng = np.random.default_rng(_seed_from_text(f"fc:{route_id}:{origin_date.isoformat()}"))
    # Scale spread monotonically with quantile so higher quantiles never fall below point forecast
    z_factor = max(0.0, (quantile - 0.50) * 2.6)

    rows = []
    for step in range(1, horizon + 1):
        d = origin_date + timedelta(days=step)
        weekend_boost = 1.20 if d.weekday() >= 4 else 0.95
        trend = 1.0 + 0.002 * step
        noise = float(rng.normal(0.0, base * 0.03))
        point = max(0, int(round(base * weekend_boost * trend + noise)))
        spread = int(round(point * (0.06 + 0.18 * z_factor)))
        q_val = max(point, point + spread)
        rows.append(
            {
                "date": d,
                "route_id": route_id,
                "point_forecast": point,
                "quantile_forecast": q_val,
                "quantile": quantile,
                "is_mock": True,
            }
        )
    return pd.DataFrame(rows)


def plan_operations(
    forecast_df: pd.DataFrame,
    occupancy_target: float = 0.75,
    bus_capacity: int = 40,
    demand_col: str = "quantile_forecast",
) -> dict[str, Any]:
    """Mock contract for P2: converts passenger forecast into daily dispatches, fleet, and drivers."""
    if forecast_df.empty:
        raise ValueError("El DataFrame de pronóstico está vacío.")
    if not (0.30 <= occupancy_target <= 1.0):
        raise ValueError("La ocupación objetivo debe estar entre 0.30 y 1.00.")
    if bus_capacity <= 0:
        raise ValueError("La capacidad del vehículo debe ser positiva.")

    effective_capacity = bus_capacity * occupancy_target
    schedule = forecast_df[["date", "route_id", "point_forecast", "quantile_forecast"]].copy()
    schedule["dispatches"] = schedule[demand_col].apply(
        lambda pax: int(math.ceil(pax / effective_capacity))
    )
    # Assume each vehicle can cover ~1.5 round-trip/dispatch slots per day on intercity corridors
    schedule["vehicles_required"] = schedule["dispatches"].apply(
        lambda disp: int(math.ceil(disp / 1.5))
    )
    # 1.25 driver-to-vehicle ratio to respect rest/shift regulations (Resolución 202030400 / Mintransporte)
    schedule["drivers_required"] = schedule["vehicles_required"].apply(
        lambda veh: int(math.ceil(veh * 1.25))
    )

    return {
        "schedule": schedule,
        "total_passengers_point": int(schedule["point_forecast"].sum()),
        "total_passengers_quantile": int(schedule["quantile_forecast"].sum()),
        "avg_daily_dispatches": round(float(schedule["dispatches"].mean()), 1),
        "peak_day_vehicles": int(schedule["vehicles_required"].max()),
        "peak_day_drivers": int(schedule["drivers_required"].max()),
        "occupancy_target": occupancy_target,
        "is_mock": True,
    }


def _build_mock_gradcam(img: Image.Image, alert_bool: bool, seed: int) -> bytes:
    """Create a synthetic Grad-CAM attention overlay for UI demonstration."""
    rgb = img.convert("RGB").resize((320, 240))
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    rng = np.random.default_rng(seed)
    cx = int(rng.integers(90, 230))
    cy = int(rng.integers(70, 170))
    color_outer = (239, 68, 68, 95) if alert_bool else (16, 185, 129, 85)
    color_inner = (245, 158, 11, 130) if alert_bool else (59, 130, 246, 110)

    draw.ellipse((cx - 65, cy - 55, cx + 65, cy + 55), fill=color_outer)
    draw.ellipse((cx - 32, cy - 28, cx + 32, cy + 28), fill=color_inner)

    blended = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    return buf.getvalue()


def generate_sample_driver_image() -> bytes:
    """Generate a lightweight synthetic cabin image for demo testing without external files."""
    img = Image.new("RGB", (320, 240), color=(30, 41, 59))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 150, 300, 230), fill=(51, 65, 85))
    draw.ellipse((110, 60, 210, 160), outline=(148, 163, 184), width=6)
    draw.text((85, 25), "Muestra cabina (Demo)", fill=(226, 232, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def classify_driver(image_bytes: bytes, alert_threshold: float = 0.50) -> dict[str, Any]:
    """Mock contract for P3: classifies driver behavior and returns probabilities + Grad-CAM."""
    if not image_bytes:
        raise ValueError("La imagen está vacía.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("La imagen supera el límite máximo permitido de 5 MB.")
    if not (0.0 <= alert_threshold <= 1.0):
        raise ValueError("El umbral de alerta debe estar entre 0.0 y 1.0.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError("El archivo no es una imagen JPG/PNG válida.") from exc

    seed = int(hashlib.sha256(image_bytes).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(len(DRIVER_CLASSES)) * 1.4)
    probs = {cls: round(float(p), 4) for cls, p in zip(DRIVER_CLASSES, raw)}
    # Ensure exact sum = 1.0
    diff = round(1.0 - sum(probs.values()), 4)
    probs[DRIVER_CLASSES[0]] = round(probs[DRIVER_CLASSES[0]] + diff, 4)

    top_class = max(probs, key=lambda k: probs[k])
    alert_prob = round(sum(probs[c] for c in ALERT_CLASSES), 4)
    alert_bool = alert_prob >= alert_threshold

    return {
        "class_name": top_class,
        "class_label": DRIVER_CLASS_LABELS[top_class],
        "class_probabilities": probs,
        "alert_probability": alert_prob,
        "alert_bool": alert_bool,
        "alert_threshold": alert_threshold,
        "gradcam_png_bytes": _build_mock_gradcam(img, alert_bool, seed),
        "is_mock": True,
    }


def recommend(
    user_id: str | None = None,
    profile: dict[str, Any] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Mock contract for P4: returns top-K Indian tourist destinations with cold-start support."""
    if k < 1 or k > len(DESTINATIONS):
        raise ValueError(f"K debe estar entre 1 y {len(DESTINATIONS)}.")

    prefs = set((profile or {}).get("preferences", []))
    is_cold_start = user_id is None or user_id not in USER_HISTORIES
    history = [] if is_cold_start else list(USER_HISTORIES[user_id])

    scored: list[dict[str, Any]] = []
    for dest in DESTINATIONS:
        if dest["destination_name"] in history:
            continue
        pref_match = dest["category"] in prefs
        bonus = 0.25 if pref_match else 0.0
        score = round(min(0.99, dest["base_popularity"] * 0.72 + bonus), 3)
        if pref_match:
            reason = f"Coincide con tu preferencia '{dest['category']}'"
        elif is_cold_start:
            reason = "Alta popularidad inicial (cold-start)"
        else:
            reason = "Afinidad con usuarios de perfil similar"

        scored.append(
            {
                "destination_id": dest["destination_id"],
                "destination_name": dest["destination_name"],
                "category": dest["category"],
                "region": dest["region"],
                "score": score,
                "reason": reason,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return {
        "user_id": user_id,
        "is_cold_start": is_cold_start,
        "history": history,
        "recommendations": scored[:k],
        "is_mock": True,
    }
