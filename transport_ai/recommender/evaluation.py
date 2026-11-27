"""Evaluación uniforme: mismos candidatos y particiones para todos los modelos."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .classical import load_knn, score_knn
from .coldstart import score_content
from .data import read_csv, write_json
from .metrics import rank_metrics

EVAL_SPLITS = ("validation", "test", "cold_test")


def load_context(processed: Path) -> dict:
    manifest, _ = read_csv(processed / "split_manifest.csv")
    catalog, _ = read_csv(processed / "catalog.csv")
    profiles, _ = read_csv(processed / "profiles.csv")
    items = {x["destination_id"] for x in catalog}
    if len(items) != len(catalog):
        raise ValueError("Catálogo con identificadores duplicados")
    known = {p["user_id"]: p for p in profiles}
    train_positive: dict[str, set[str]] = defaultdict(set)
    # Ítems que el usuario ya valoró en entrenamiento: no se deben volver a recomendar.
    observed: dict[str, set[str]] = defaultdict(set)
    heldout: dict[str, dict[str, set[str]]] = {s: defaultdict(set) for s in EVAL_SPLITS}
    all_test_pairs: set[tuple[str, str]] = set()
    for row in manifest:
        u, d, split = row["user_id"], row["destination_id"], row["split"]
        if d not in items:
            raise ValueError("Hay interacciones de un destino ausente del catálogo")
        if split == "train":
            train_positive[u].add(d)
            observed[u].add(d)
        elif split == "observed_nonpositive":
            observed[u].add(d)
        elif split in EVAL_SPLITS:
            if (u, d) in all_test_pairs:
                raise ValueError("Un par está en varios conjuntos de evaluación")
            all_test_pairs.add((u, d))
            heldout[split][u].add(d)
    if {u for u, _ in all_test_pairs if u in heldout["cold_test"]} & set(train_positive):
        raise ValueError("Fuga de usuarios cold-start a train")
    return {"manifest": manifest, "catalog": catalog, "catalog_ids": items, "profiles": known, "train": train_positive,
            "observed": observed, "truth": heldout, "popularity": Counter(d for hist in train_positive.values() for d in hist)}


def rank_ids(scores: dict[str, float], allowed: set[str], k: int) -> list[str]:
    if k < 1:
        raise ValueError("k debe ser >= 1")
    return sorted(allowed, key=lambda d: (-scores.get(d, 0.0), d))[:k]


def get_scores(context: dict, user: str, method: str, knn: dict | None) -> dict[str, float]:
    if method == "popularity":
        return dict(context["popularity"])
    if method == "item-knn":
        if knn is None:
            raise ValueError("Falta modelo item-knn guardado")
        return score_knn(knn, context["train"].get(user, set()))
    if method == "content":
        profile = context["profiles"].get(user, {})
        return score_content(context["catalog"], profile.get("preferences", ""), context["popularity"])
    raise ValueError(f"Modelo desconocido: {method}")


def compare_models(processed: Path, artifact: Path, k: int = 5, split: str = "validation") -> dict:
    if split not in EVAL_SPLITS:
        raise ValueError(f"Split debe ser uno de {EVAL_SPLITS}")
    ctx = load_context(processed)
    knn = load_knn(artifact)
    relevant_by_user = ctx["truth"][split]
    result = {"split": split, "k": k, "users_evaluated": len(relevant_by_user), "models": {},
              "note": "Evaluación offline con datos muy dispersos; hits cercanos a cero no prueban fallo de ejecución."}
    for method in ("popularity", "item-knn", "content"):
        values, recommendation_union = [], set()
        buckets = {"few": [], "many": []}
        examples = []
        for user, relevant in sorted(relevant_by_user.items()):
            seen = ctx["observed"].get(user, set())
            allowed = ctx["catalog_ids"] - seen
            scores = get_scores(ctx, user, method, knn)
            ranked = rank_ids(scores, allowed, k)
            recommendation_union.update(ranked)
            value = rank_metrics(ranked, relevant, k)
            values.append(value)
            # Segmentación basada en interacciones disponibles, nunca en test.
            bucket = "few" if len(ctx["train"].get(user, set())) < 2 else "many"
            buckets[bucket].append(value)
            if len(examples) < 5:
                examples.append({"user_id": user, "train_positive_count": len(ctx["train"].get(user, set())),
                                 "recommendations": ranked, "heldout_positive_count": len(relevant)})
        n = len(values)
        avg = lambda rows, field: sum(v[field] for v in rows) / len(rows) if rows else None
        result["models"][method] = {
            "precision_at_k": avg(values, "precision"), "recall_at_k": avg(values, "recall"),
            "ndcg_at_k": avg(values, "ndcg"),
            "catalog_coverage": len(recommendation_union) / len(ctx["catalog_ids"]) if ctx["catalog_ids"] else None,
            "segments_by_train_positive_count": {
                key: {"users": len(val), "precision_at_k": avg(val, "precision"), "recall_at_k": avg(val, "recall"), "ndcg_at_k": avg(val, "ndcg")}
                for key, val in buckets.items()
            },
            "example_rankings": examples,
        }
    return result
