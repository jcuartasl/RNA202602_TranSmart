"""Partición determinista para ranking implícito sin usar datos de test en train.

Opiniones de 4/5 = señal positiva (SUPUESTO, por validar). 1-3 quedan
registradas, pero no son positivos. Una misma pareja usuario-destino jamás
se separa entre conjuntos. No hay evaluación temporal global porque reviews
no contiene fechas: las fechas de visita por sí solas no bastan.
"""
from __future__ import annotations

import random
from math import ceil
from collections import Counter, defaultdict
from pathlib import Path

from .data import read_csv, write_csv, write_json


def build_split(processed: Path, seed: int = 42, cold_fraction: float = 0.15, positive_at: int = 4) -> dict:
    if not 0 <= cold_fraction < 1:
        raise ValueError("cold_fraction debe estar en [0,1)")
    if not 1 <= positive_at <= 5:
        raise ValueError("positive_at debe estar entre 1 y 5")
    interactions, _ = read_csv(processed / "interactions.csv")
    profiles, _ = read_csv(processed / "profiles.csv")
    ids = {row["user_id"] for row in profiles}
    groups = defaultdict(list)
    for row in interactions:
        groups[(row["user_id"], row["destination_id"])].append(int(row["rating"]))

    positive_pairs = defaultdict(list)
    grouped = {}
    conflicted = []
    for (u, d), ratings in sorted(groups.items()):
        positive = any(r >= positive_at for r in ratings)
        negative = any(r < positive_at for r in ratings)
        if positive and negative:
            conflicted.append((u, d))
            grouped[(u, d)] = "excluded_conflict"
        elif positive:
            positive_pairs[u].append(d)
        else:
            grouped[(u, d)] = "observed_nonpositive"

    rng = random.Random(seed)
    cold_candidates = sorted(positive_pairs)
    rng.shuffle(cold_candidates)
    n_cold = min(max(0, len(cold_candidates) - 1), ceil(cold_fraction * len(cold_candidates)))
    cold_users = set(cold_candidates[:n_cold])

    for u in sorted(positive_pairs):
        destinations = sorted(positive_pairs[u])
        rng.shuffle(destinations)
        if u in cold_users:
            grouped.update({(u, d): "cold_test" for d in destinations})
            continue
        if len(destinations) >= 3:
            grouped[(u, destinations[0])] = "test"
            grouped[(u, destinations[1])] = "validation"
            for d in destinations[2:]:
                grouped[(u, d)] = "train"
        elif len(destinations) == 2:
            grouped[(u, destinations[0])] = "test"
            grouped[(u, destinations[1])] = "train"
        else:
            grouped[(u, destinations[0])] = "train"

    # Reservar valoraciones explicitamente NO positivas (1-3) para una BCE
    # de validacion real, sin tratar ausencias como negativos ni tocar test.
    # Proporcion 1:1 con los positivos warm de validacion; se priorizan los
    # mismos usuarios de validacion para no introducir cold-start inadvertido.
    val_users = {u for (u, _d), split in grouped.items() if split == "validation"}
    n_val_positive = sum(split == "validation" for split in grouped.values())
    eligible_negative = sorted(pair for pair, split in grouped.items()
                               if split == "observed_nonpositive" and pair[0] not in cold_users)
    rng_negative = random.Random(seed + 7919)
    rng_negative.shuffle(eligible_negative)
    eligible_negative.sort(key=lambda pair: pair[0] not in val_users)
    reserved_negative = eligible_negative[:n_val_positive]
    for pair in reserved_negative:
        grouped[pair] = "validation_nonpositive"

    # Todos los registros del usuario cold-start se retienen fuera del entrenamiento.
    for pair, part in list(grouped.items()):
        if pair[0] in cold_users and part == "observed_nonpositive":
            grouped[pair] = "cold_nonpositive_holdout"

    if len(groups) != len(grouped):
        raise AssertionError("Algún par no recibió partición")
    rows = [
        {"user_id": u, "destination_id": d, "split": grouped[(u, d)], "positive": str(grouped[(u, d)] in {"train", "validation", "test", "cold_test"}).lower()}
        for (u, d) in sorted(grouped)
    ]
    write_csv(processed / "split_manifest.csv", ["user_id", "destination_id", "split", "positive"], rows)
    counts = Counter(r["split"] for r in rows)
    training_users = {r["user_id"] for r in rows if r["split"] == "train"}
    if training_users & cold_users:
        raise AssertionError("Fuga: usuario cold-start en entrenamiento")
    report = {
        "seed": seed,
        "cold_fraction_requested": cold_fraction,
        "positive_at": positive_at,
        "definition": "4-5 = preferencia positiva (supuesto experimental, no clic observado); 1-3 no son positivos; conflictos excluidos",
        "split_method": "holdout por usuario para positivos y usuarios completos para cold-start; NO temporal global",
        "counts_by_split": dict(sorted(counts.items())),
        "cold_users": len(cold_users),
        "cold_users_ids_file": "cold_user_ids.csv",
        "users_with_no_positive": len(ids - set(positive_pairs)),
        "conflicted_pairs": len(conflicted),
        "validation_bce_design": "Solo etiquetas explicitas 4-5 (positivas) y 1-3 (no positivas); "
                                  "las negativas de validacion se excluyen del entrenamiento; "
                                  "submuestreo 1:1 si hay suficientes candidatas warm",
        "validation_bce_positive_count": n_val_positive,
        "validation_bce_nonpositive_count": len(reserved_negative),
        "warnings": [
            "Hay pocas interacciones por usuario: la validación warm puede ser pequeña.",
            "No considerar ítems no observados como negativos verdaderos.",
            "Un historial incompleto y dos fuentes con fechas distintas impiden afirmar causalidad temporal.",
        ],
    }
    write_csv(processed / "cold_user_ids.csv", ["user_id"], [{"user_id": u} for u in sorted(cold_users)])
    write_json(processed / "split_report.json", report)
    return report
