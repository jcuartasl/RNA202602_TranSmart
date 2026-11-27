"""Filtrado colaborativo clásico: similitud coseno ítem-ítem por co-ocurrencia.

Se ajusta únicamente con las interacciones positivas etiquetadas 'train'.
No usa Popularity del CSV original ni observa validación/test para ajustar.
El respaldo de popularidad es explícito cuando no hay soporte colaborativo.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from math import sqrt
from pathlib import Path

from .data import read_csv, write_json


def train_item_knn(processed: Path, artifact: Path, neighbors: int = 20) -> dict:
    if neighbors < 1:
        raise ValueError("neighbors debe ser >= 1")
    rows, _ = read_csv(processed / "split_manifest.csv")
    catalog, _ = read_csv(processed / "catalog.csv")
    items = {row["destination_id"] for row in catalog}
    user_history: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] == "train":
            if row["destination_id"] not in items:
                raise ValueError("Destinos de train faltan en el catálogo")
            user_history[row["user_id"]].add(row["destination_id"])

    popularity = Counter(item for hist in user_history.values() for item in hist)
    cooc = Counter()
    for history in user_history.values():
        for pair in combinations(sorted(history), 2):
            cooc[pair] += 1

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (item_a, item_b), common in cooc.items():
        similarity = common / sqrt(popularity[item_a] * popularity[item_b])
        adjacency[item_a].append((item_b, similarity))
        adjacency[item_b].append((item_a, similarity))

    links = {
        item: [
            {"destination_id": other, "similarity": value}
            for other, value in sorted(adjacency.get(item, []), key=lambda x: (-x[1], x[0]))[:neighbors]
        ]
        for item in sorted(items)
    }
    payload = {
        "model": "item_knn_cosine_train_only",
        "neighbors": neighbors,
        "train_users": len(user_history),
        "train_positive_interactions": sum(map(len, user_history.values())),
        "cooccurrence_item_pairs": len(cooc),
        "items_with_neighbors": sum(bool(v) for v in adjacency.values()),
        "positive_counts": dict(sorted(popularity.items())),
        "neighbors_by_item": links,
        "catalog_ids": sorted(items),
        "fitted_splits": ["train"],
        "note": "Si faltan vecinos para un usuario, se utiliza popularidad del entrenamiento como respaldo.",
    }
    write_json(artifact, payload)
    return payload


def load_knn(artifact: Path) -> dict:
    if not artifact.exists():
        raise FileNotFoundError(f"Falta el modelo guardado: {artifact}. Ejecuta 'fit-classical'.")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("model") != "item_knn_cosine_train_only":
        raise ValueError("Artefacto de filtrado colaborativo inválido")
    return payload


def score_knn(model: dict, seen_positive: set[str]) -> dict[str, float]:
    """Puntaje colaborativo; el respaldo usa solo popularidad de train."""
    scores: dict[str, float] = defaultdict(float)
    for liked in seen_positive:
        for related in model["neighbors_by_item"].get(liked, []):
            scores[related["destination_id"]] += related["similarity"]
    for item, count in model["positive_counts"].items():
        # Desempate determinista; no supera una similitud colaborativa positiva.
        scores[item] += 0.0001 * count
    return dict(scores)
