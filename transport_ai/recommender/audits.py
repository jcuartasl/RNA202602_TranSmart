"""Auditoría de entidades repetidas en el catálogo de recomendaciones (P4).

IMPORTANTE: no fusiona registros ni altera IDs/splits: solo detecta posibles
colisiones semánticas y las documenta para decidir con evidencia.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .data import read_csv, write_json


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def catalog_audit(processed: Path, output: Path | None = None) -> dict:
    rows, _ = read_csv(processed / "catalog.csv")
    by_name, by_name_state, by_name_state_type = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        n, state, typ = (normalize(row.get(x, "")) for x in ("destination_name", "state", "type"))
        by_name[n].append(row)
        by_name_state[(n, state)].append(row)
        by_name_state_type[(n, state, typ)].append(row)

    def summarize(groups: dict, limit: int = 12) -> dict:
        repeats = sorted((v for v in groups.values() if len(v) > 1), key=lambda v: (-len(v), v[0]["destination_name"]))
        examples = []
        for group in repeats[:limit]:
            examples.append({"normalized_name": normalize(group[0]["destination_name"]),
                             "count": len(group),
                             "records": [
                                 {"destination_id": r["destination_id"], "destination_name": r["destination_name"],
                                  "state": r["state"], "type": r["type"]} for r in group[:10]
                             ], "truncated_records": max(0, len(group) - 10)})
        return {"repeated_groups": len(repeats),
                "excess_ids": sum(len(g) - 1 for g in repeats),
                "ids_in_repeated_groups": sum(len(g) for g in repeats),
                "examples": examples}

    result = {"catalog_rows": len(rows), "unique_ids": len({r["destination_id"] for r in rows}),
              "unique_normalized_names": len(by_name),
              "same_name": summarize(by_name), "same_name_and_state": summarize(by_name_state),
              "same_name_state_and_type": summarize(by_name_state_type),
              "warning": ("Los nombres repetidos no demuestran por sí solos entidades idénticas. "
                          "NO fusionar IDs, modificar splits ni reentrenar automáticamente. "
                          "Si se establece una política de deduplicación, usarla también en todos los modelos de evaluación.")}
    if output:
        write_json(output, result)
    return result


def ranking_duplicate_names(recommendations: list[dict]) -> dict:
    by = defaultdict(list)
    for row in recommendations:
        by[normalize(row["destination_name"])].append(row["destination_id"])
    duplicates = {key: ids for key, ids in by.items() if len(ids) > 1}
    return {"ranking_items": len(recommendations),
            "distinct_names": len(by), "duplicated_names": duplicates}
