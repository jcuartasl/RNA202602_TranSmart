"""Baseline de contenido para cold-start usando Preferences y Type disponibles.

No usa la columna Popularity del CSV: las cuentas provienen solo de train.
La correspondencia es explícita, no una red neuronal entrenada.
"""
from __future__ import annotations

import re
from collections import Counter

PREFERENCE_ALIASES = {
    "beach": "beach", "beaches": "beach", "playa": "beach", "playas": "beach",
    "historical": "historical", "history": "historical", "historic": "historical",
    "nature": "nature", "naturaleza": "nature",
    "city": "city", "cities": "city", "ciudad": "city",
    "adventure": "adventure", "aventura": "adventure",
}


def parse_preferences(value: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if value is None:
        return set()
    tokens = re.split(r"[,;/|]", value) if isinstance(value, str) else value
    return {PREFERENCE_ALIASES[x.strip().lower()] for x in tokens if x.strip().lower() in PREFERENCE_ALIASES}


def score_content(catalog: list[dict], preferences: str | list[str] | None, train_popularity: dict[str, int]) -> dict[str, float]:
    preferred = parse_preferences(preferences)
    max_pop = max(train_popularity.values(), default=0)
    scores = {}
    for row in catalog:
        destination_id = row["destination_id"]
        preferred_type = str(row["type"]).strip().lower() in preferred
        # Dos estrategias claramente distintas: popularidad global y coincidencia de preferencias.
        # El término de popularidad es SOLO desempate entre destinos del mismo grupo.
        normalized_pop = train_popularity.get(destination_id, 0) / max(1, max_pop)
        scores[destination_id] = (1.0 if preferred_type else 0.0) + 0.01 * normalized_pop
    return scores
