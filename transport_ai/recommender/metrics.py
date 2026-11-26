"""Métricas de ranking con denominadores explícitos."""
from __future__ import annotations

from math import log2


def rank_metrics(recommended: list[str], relevant: set[str], k: int) -> dict[str, float]:
    if k < 1:
        raise ValueError("k debe ser positivo")
    top = recommended[:k]
    if len(top) != len(set(top)):
        raise ValueError("El ranking contiene destinos duplicados")
    hits = sum(d in relevant for d in top)
    dcg = sum((1 / log2(i + 2)) for i, d in enumerate(top) if d in relevant)
    ideal = sum((1 / log2(i + 2)) for i in range(min(k, len(relevant))))
    return {
        "precision": hits / k,
        "recall": hits / len(relevant) if relevant else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
    }
