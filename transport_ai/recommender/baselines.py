"""Primer baseline obligatorio: popularidad aprendida SOLO del train.

No utiliza 'Popularity' del dataset: su procedencia temporal es desconocida.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import read_csv, write_json
from .metrics import rank_metrics


def popularity_scores(processed: Path) -> Counter:
    parts, _ = read_csv(processed / "split_manifest.csv")
    return Counter(row["destination_id"] for row in parts if row["split"] == "train")


def evaluate_popularity(processed: Path, k: int = 5) -> dict:
    if k < 1:
        raise ValueError("k debe ser >= 1")
    catalog, _ = read_csv(processed / "catalog.csv")
    parts, _ = read_csv(processed / "split_manifest.csv")
    items = {row["destination_id"] for row in catalog}
    scores = popularity_scores(processed)
    seen = defaultdict(set)
    heldout = {"validation": defaultdict(set), "test": defaultdict(set), "cold_test": defaultdict(set)}
    for row in parts:
        user, dest, part = row["user_id"], row["destination_id"], row["split"]
        if part in ("train", "observed_nonpositive"):
            seen[user].add(dest)
        elif part in heldout:
            heldout[part][user].add(dest)

    def ranking(user: str) -> list[str]:
        return sorted(items - seen[user], key=lambda d: (-scores[d], d))[:k]

    result = {"model": "popularity_train_only", "k": k, "segments": {}, "zero_support_items": sum(scores[d] == 0 for d in items)}
    for segment, truth in heldout.items():
        metrics = []
        recommended_all = set()
        for user, relevant in truth.items():
            recommended = ranking(user)
            recommended_all.update(recommended)
            metrics.append(rank_metrics(recommended, relevant, k))
        result["segments"][segment] = {
            "users_evaluated": len(metrics),
            "precision_at_k": sum(m["precision"] for m in metrics) / len(metrics) if metrics else None,
            "recall_at_k": sum(m["recall"] for m in metrics) / len(metrics) if metrics else None,
            "ndcg_at_k": sum(m["ndcg"] for m in metrics) / len(metrics) if metrics else None,
            "catalog_coverage": len(recommended_all) / len(items) if items else None,
        }
    write_json(processed / "baseline_popularity_metrics.json", result)
    return result
