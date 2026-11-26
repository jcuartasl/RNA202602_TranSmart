"""Escenario de campaña bajo supuestos explícitos, NO conversiones observadas.

No se interpreta la salida sigmoide de la red como una probabilidad calibrada
de compra. Las tasas de exposición/conversión se suministran externamente.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .coldstart import parse_preferences
from .data import read_csv, write_json


def campaign_scenario(processed: Path, output: Path, *, destination_type: str = "beach",
                      exposure_rate: float = 0.5, assumed_conversion_rate: float = 0.03) -> dict:
    if not (0 <= exposure_rate <= 1 and 0 <= assumed_conversion_rate <= 1):
        raise ValueError("Las tasas deben estar entre 0 y 1")
    destination_type = destination_type.strip().lower()
    catalog, _ = read_csv(processed / "catalog.csv")
    profiles, _ = read_csv(processed / "profiles.csv")
    manifest, _ = read_csv(processed / "split_manifest.csv")
    available = [r for r in catalog if r["type"].lower() == destination_type]
    if not available:
        raise ValueError(f"No existe Type={destination_type} en el catálogo")
    train_counts = Counter(r["destination_id"] for r in manifest if r["split"] == "train")
    chosen = sorted(available, key=lambda r: (-train_counts[r["destination_id"]], r["destination_id"]))[0]
    audience = [r for r in profiles if destination_type in parse_preferences(r.get("preferences", ""))]
    estimated_exposed = len(audience) * exposure_rate
    report = {
        "destination": {"id": chosen["destination_id"], "name": chosen["destination_name"],
                        "type": chosen["type"], "state": chosen["state"]},
        "selection_rule": "Destino del tipo seleccionado con más valoraciones positivas EN TRAIN; desempate por ID",
        "audience_rule": "Usuarios cuyo campo Preferences incluye ese tipo",
        "eligible_profiles": len(audience),
        "exposure_rate_assumed": exposure_rate,
        "conversion_rate_assumed": assumed_conversion_rate,
        "expected_exposures": estimated_exposed,
        "expected_conversions": estimated_exposed * assumed_conversion_rate,
        "expected_conversions_rounded": round(estimated_exposed * assumed_conversion_rate),
        "sensitivity_expected_conversions": {
            "half_conversion": estimated_exposed * assumed_conversion_rate / 2,
            "double_conversion_capped_at_100pct": estimated_exposed * min(1.0, assumed_conversion_rate * 2),
        },
        "limitations": [
            "No existen etiquetas de compras, clics ni conversiones para calibrar esta estimación",
            "El valor de salida de la red NO es una probabilidad de compra validada",
            "Las cifras son escenarios hipotéticos, no una promesa comercial",
            "El catálogo de destinos de India no equivale a las rutas colombianas del módulo de flota",
        ],
    }
    write_json(output, report)
    return report
