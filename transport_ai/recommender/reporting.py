"""Informe P4: métricas para distintos K, segmentos, diversidad y ejemplos.

La evaluación aplica IGUAL conjunto de candidatos y mismo ground truth a
popularidad, kNN, contenido y red dos torres. NO entrena ni altera splits.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .classical import load_knn
from .data import read_csv, write_csv, write_json
from .evaluation import get_scores, load_context, rank_ids
from .metrics import rank_metrics
from .neural import load_checkpoint, score_catalog

METHODS = ("popularity", "item-knn", "content", "neural")


def _avg(rows: list[dict], key: str) -> float | None:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else None


def _concentration(recommended: Counter) -> dict:
    total = sum(recommended.values())
    if not total:
        return {"hhi": None, "top_5_share": None, "distinct_destinations": 0, "total_recommendations": 0}
    shares = sorted((n / total for n in recommended.values()), reverse=True)
    return {"hhi": sum(p * p for p in shares),
            "top_5_share": sum(shares[:5]),
            "distinct_destinations": len(shares),
            "total_recommendations": total}


def _choose_examples(ctx: dict, users: list[str], n: int = 5) -> list[str]:
    """Equilibra combinaciones de preferencias y actividad sin consultar test."""
    selected, used = [], set()
    for user in users:
        preference = ctx["profiles"].get(user, {}).get("preferences", "")
        activity = "muchas" if len(ctx["train"].get(user, set())) >= 2 else "pocas"
        key = (preference, activity)
        if key not in used:
            selected.append(user)
            used.add(key)
        if len(selected) >= n:
            return selected
    for user in users:
        if user not in selected:
            selected.append(user)
        if len(selected) >= n:
            return selected
    return selected


def _training_frequency(ctx: dict) -> dict:
    users = ctx["profiles"]
    catalog = ctx["catalog_ids"]
    positive = ctx["train"]
    by_user = {u: len(positive.get(u, set())) for u in users}
    item_counts = Counter(d for items in positive.values() for d in items)
    by_item = {d: item_counts[d] for d in catalog}
    return {"positive_count_per_user": by_user, "positive_count_per_destination": by_item,
            "train_positive_interactions": sum(by_user.values()),
            "train_users_with_zero_positive": sum(n == 0 for n in by_user.values()),
            "train_destinations_with_zero_positive": sum(n == 0 for n in by_item.values())}


def generate_report(processed: Path, knn_artifact: Path, neural_checkpoint: Path,
                    output: Path, *, split: str = "validation", ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict:
    if split not in ("validation", "test", "cold_test"):
        raise ValueError("split debe ser validation, test o cold_test")
    if not ks or min(ks) < 1:
        raise ValueError("Todos los K deben ser enteros positivos")
    ks = tuple(sorted(set(int(k) for k in ks)))
    ctx = load_context(processed)
    knn = load_knn(knn_artifact)
    model, bundle, ckpt = load_checkpoint(neural_checkpoint, processed)
    users = sorted(ctx["truth"][split])
    if not users:
        raise ValueError("El conjunto de evaluación no tiene usuarios positivos")
    catalog_by_id = {r["destination_id"]: r for r in ctx["catalog"]}
    buckets = {"few": [u for u in users if len(ctx["train"].get(u, set())) < 2],
               "many": [u for u in users if len(ctx["train"].get(u, set())) >= 2]}
    metric_rows = []
    examples = {}
    picks = _choose_examples(ctx, users)
    for method in METHODS:
        scores_by_user = {}
        for user in users:
            if method == "neural":
                scores = score_catalog(model, bundle, user)
            else:
                scores = get_scores(ctx, user, method, knn)
            allowed = ctx["catalog_ids"] - ctx["observed"].get(user, set())
            scores_by_user[user] = rank_ids(scores, allowed, max(ks))
        for k in ks:
            values, recommendation_counts = [], Counter()
            segment_values = {"few": [], "many": []}
            for user in users:
                recommended = scores_by_user[user][:k]
                relevant = ctx["truth"][split][user]
                met = rank_metrics(recommended, relevant, k)
                values.append(met)
                segment_values["few" if user in buckets["few"] else "many"].append(met)
                recommendation_counts.update(recommended)
            concentration = _concentration(recommendation_counts)
            metric_rows.append({"split": split, "model": method, "k": k,
                                "users_evaluated": len(users),
                                "precision_at_k": _avg(values, "precision"),
                                "recall_at_k": _avg(values, "recall"),
                                "ndcg_at_k": _avg(values, "ndcg"),
                                "catalog_coverage": len(recommendation_counts) / len(ctx["catalog_ids"]),
                                "hhi": concentration["hhi"],
                                "top_5_share": concentration["top_5_share"],
                                "distinct_destinations": concentration["distinct_destinations"],
                                "few_users": len(buckets["few"]), "many_users": len(buckets["many"]),
                                "few_ndcg": _avg(segment_values["few"], "ndcg"),
                                "many_ndcg": _avg(segment_values["many"], "ndcg")})
        for user in picks:
            profile = ctx["profiles"][user]
            hist = [catalog_by_id[d]["destination_name"] for d in sorted(ctx["train"].get(user, set()))]
            ent = examples.setdefault(user, {"user_id": user, "preferences": profile.get("preferences", ""),
                                              "available_train_history": hist,
                                              "train_positive_count": len(hist),
                                              "models": {}})
            recs = []
            for d in scores_by_user[user][:5]:
                row = catalog_by_id[d]
                recs.append({"destination_id": d, "destination_name": row["destination_name"],
                             "state": row["state"], "type": row["type"]})
            ent["models"][method] = recs
    frequency = _training_frequency(ctx)
    output.mkdir(parents=True, exist_ok=True)
    columns = list(metric_rows[0])
    write_csv(output / f"m3_metrics_vs_k_{split}.csv", columns, metric_rows)
    write_json(output / f"m3_report_{split}.json",
               {"split": split, "metrics": metric_rows, "examples": list(examples.values()),
                "training_frequency": frequency,
                "warnings": [
                    "No se alteraron modelos, datos ni particiones en este reporte",
                    "Las preferencias de usuarios cold-start se consideran datos de onboarding disponibles",
                    "Las valoraciones positivas retenidas nunca se muestran como historial",
                    "La cobertura se calcula por ID; nombres físicamente repetidos pueden inflarla",
                    "La evaluación con pocos positivos por usuario tiene incertidumbre elevada",
                    "Las probabilidades sigmoid no son probabilidades calibradas de compra",
                ]})
    return {"split": split, "users_evaluated": len(users), "ks": list(ks),
            "model_seed_selected": ckpt["seed"],
            "metrics_by_k": {str(k): {r["model"]: {"ndcg_at_k": r["ndcg_at_k"],
                                                     "precision_at_k": r["precision_at_k"],
                                                     "recall_at_k": r["recall_at_k"],
                                                     "coverage": r["catalog_coverage"]}
                                   for r in metric_rows if r["k"] == k}
                             for k in ks},
            "rows_written": len(metric_rows),
            "metrics_csv": str(output / f"m3_metrics_vs_k_{split}.csv"),
            "report_json": str(output / f"m3_report_{split}.json"),
            "metrics_at_5": {r["model"]: {"ndcg_at_5": r["ndcg_at_k"], "coverage": r["catalog_coverage"]}
                             for r in metric_rows if r["k"] == 5}}
