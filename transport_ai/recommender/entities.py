"""Unidad de recomendación P4: comprobar y, opcionalmente, normalizar entidades.

El CSV de Kaggle posee IDs distintos que pueden representar el mismo lugar. Este
módulo produce una **versión alternativa** de los datos, sin modificar el
procesamiento ni los checkpoints previamente generados. Cualquier experimento
con los IDs originales y con entidades normalizadas se reporta por separado.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .audits import normalize
from .data import read_csv, write_csv, write_json


def entity_key(row: dict[str, str]) -> tuple[str, str, str]:
    """Clave conservadora: nombre, estado y tipo simultáneamente iguales."""
    return tuple(normalize(row.get(field, "")) for field in ("destination_name", "state", "type"))


def prepare_entities(source: Path, target: Path, *, positive_at: int = 4) -> dict:
    """Copia perfil; transforma IDs al lugar lógico; preserva valoración original.

    No hay ningún entrenamiento y no lee test para aprender características.
    La equivalencia se establece con metadatos del catálogo conocidos antes de
    separar interacciones. No es prueba de identidad física, sino criterio de
    distinción que permite este dataset.
    """
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("La salida normalizada debe ser distinta al directorio original")
    if not 1 <= positive_at <= 5:
        raise ValueError("positive_at debe estar entre 1 y 5")

    catalog, _ = read_csv(source / "catalog.csv")
    profiles, _ = read_csv(source / "profiles.csv")
    interactions, _ = read_csv(source / "interactions.csv")
    keys = sorted({entity_key(row) for row in catalog})
    if any(not all(key) for key in keys):
        raise ValueError("No se puede normalizar entidad sin nombre, estado y tipo")

    index = {key: f"place_{i:04d}" for i, key in enumerate(keys, 1)}
    grouped = defaultdict(list)
    for row in catalog:
        grouped[entity_key(row)].append(row)

    normalized_catalog = []
    id_to_entity = {}
    id_rows = []
    for key in keys:
        group = sorted(grouped[key], key=lambda r: r["destination_id"])
        representative = group[0]
        cid = index[key]
        normalized_catalog.append({
            "destination_id": cid,
            "destination_name": representative["destination_name"],
            "state": representative["state"],
            "type": representative["type"],
            "best_time_to_visit": representative["best_time_to_visit"],
        })
        for row in group:
            did = row["destination_id"]
            if did in id_to_entity:
                raise ValueError(f"ID repetido en catálogo de origen: {did}")
            id_to_entity[did] = cid
            id_rows.append({"original_destination_id": did, "entity_destination_id": cid,
                            "destination_name": row["destination_name"], "state": row["state"],
                            "type": row["type"]})

    normalized_interactions = []
    by_user_entity = defaultdict(list)
    for row in interactions:
        old_id = row["destination_id"]
        if old_id not in id_to_entity:
            raise ValueError(f"Interacción sin catálogo: {old_id}")
        new = dict(row)
        new["destination_id"] = id_to_entity[old_id]
        normalized_interactions.append(new)
        by_user_entity[(new["user_id"], new["destination_id"])].append(int(new["rating"]))

    conflict_groups = [pair for pair, ratings in by_user_entity.items()
                       if any(r >= positive_at for r in ratings)
                       and any(r < positive_at for r in ratings)]
    # build_split excluirá estos grupos de etiquetas contradictorias enteros.
    # Se preservan las valoraciones crudas y se genera un informe sin identificadores
    # personales para explicar cuántos conflictos aparecieron tras normalizar.
    write_csv(target / "catalog.csv",
              ["destination_id", "destination_name", "state", "type", "best_time_to_visit"],
              normalized_catalog)
    write_csv(target / "profiles.csv",
              ["user_id", "preferences", "number_of_adults", "number_of_children"],
              profiles)
    write_csv(target / "interactions.csv",
              ["user_id", "destination_id", "source", "rating", "visit_date"],
              normalized_interactions)
    write_csv(target / "entity_id_map.csv",
              ["original_destination_id", "entity_destination_id", "destination_name", "state", "type"],
              sorted(id_rows, key=lambda r: r["original_destination_id"]))
    stats = Counter(id_to_entity.values())
    report = {
        "original_catalog_rows": len(catalog),
        "original_distinct_destination_ids": len(id_to_entity),
        "logical_entities_by_name_state_type": len(normalized_catalog),
        "records_per_entity": dict(sorted(stats.items())),
        "source_events": len(interactions),
        "destination_user_pairs_after_grouping": len(by_user_entity),
        "destination_user_pairs_with_multiple_events": sum(len(v) > 1 for v in by_user_entity.values()),
        "entity_user_pairs_conflicting_positive_nonpositive": len(conflict_groups),
        "policy": "Equivalencia conservadora por nombre+estado+tipo normalizados; no implica identidad geográfica verificada",
        "warning": [
            "No comparar directamente NDCG ni cobertura por ID original con resultados de entidad normalizada",
            "Cinco entidades distinguibles implican que top-5 puede agotar el catálogo: reportar K=1,2,3",
            "Seleccionar hiperparámetros solo mediante validación de la nueva partición",
            "Reentrenar los baselines y las 30 semillas: los pesos anteriores NO son válidos para IDs nuevos",
            "El test original ya fue inspeccionado; explicitarlo como limitación de independencia experimental",
            "Naturaleza real/sintética del dataset pendiente de verificar en el autor de Kaggle",
        ],
    }
    write_json(target / "entity_preparation_report.json", report)
    return report


def report_semantic_overlap(processed: Path, output: Path | None = None) -> dict:
    """Detecta cuándo train y holdout contienen distintos IDs del mismo lugar."""
    catalog, _ = read_csv(processed / "catalog.csv")
    manifest, _ = read_csv(processed / "split_manifest.csv")
    mapping = {row["destination_id"]: entity_key(row) for row in catalog}
    by = defaultdict(lambda: defaultdict(set))
    for r in manifest:
        if r["destination_id"] not in mapping:
            raise ValueError("Manifest con destino desconocido")
        by[r["user_id"]][r["split"]].add(mapping[r["destination_id"]])
    stats = {}
    for split in ("validation", "test", "cold_test"):
        users = [u for u, b in by.items() if b[split]]
        seen_splits = ("train", "observed_nonpositive")
        overlaps = 0
        heldout = 0
        already_seen = 0
        for u in users:
            b = by[u]
            seen = set().union(*(b[s] for s in seen_splits))
            heldout += len(b[split])
            already_seen += len(seen & b[split])
            overlaps += bool(seen & b[split])
        stats[split] = {"users": len(users),
                        "users_with_same_logical_place_in_seen_and_holdout": overlaps,
                        "heldout_distinct_logical_places": heldout,
                        "heldout_logical_places_already_seen": already_seen}
    result = {"catalog_id_count": len(mapping), "logical_catalog_size": len(set(mapping.values())),
              "overlap_by_split": stats,
              "note": "Auditoría de fuga semántica; no modifica ni reentrena modelos"}
    if output is not None:
        write_json(output, result)
    return result
