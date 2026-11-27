"""Carga, auditoría y exportación reproducible del dataset de India (P4).

Sin dependencias externas. Nunca exporta Name, Email, Gender ni ReviewText.
Se preservan IDs como cadenas; los CSV originales quedan fuera del repositorio.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

FILES = {
    "destinations": "Expanded_Destinations.csv",
    "reviews": "Final_Updated_Expanded_Reviews.csv",
    "history": "Final_Updated_Expanded_UserHistory.csv",
    "users": "Final_Updated_Expanded_Users.csv",
}
REQUIRED = {
    "destinations": ("DestinationID", "Name", "State", "Type", "Popularity", "BestTimeToVisit"),
    "reviews": ("ReviewID", "DestinationID", "UserID", "Rating", "ReviewText"),
    "history": ("HistoryID", "UserID", "DestinationID", "VisitDate", "ExperienceRating"),
    "users": ("UserID", "Name", "Email", "Preferences", "Gender", "NumberOfAdults", "NumberOfChildren"),
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
        ]
    return rows, columns


def load_all(source: Path) -> dict[str, list[dict[str, str]]]:
    datasets: dict[str, list[dict[str, str]]] = {}
    for name, filename in FILES.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Falta el archivo obligatorio: {path}")
        rows, columns = read_csv(path)
        missing = set(REQUIRED[name]) - set(columns)
        if missing:
            raise ValueError(f"{filename}: columnas faltantes {sorted(missing)}")
        datasets[name] = rows
    return datasets


def _valid_date(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _rating(value: str) -> int | None:
    try:
        number = float(value)
        if number.is_integer() and 1 <= number <= 5:
            return int(number)
    except ValueError:
        pass
    return None


def audit(datasets: dict[str, list[dict[str, str]]]) -> dict:
    users = datasets["users"]
    destinations = datasets["destinations"]
    uid = [row["UserID"] for row in users if row["UserID"]]
    did = [row["DestinationID"] for row in destinations if row["DestinationID"]]
    valid_users, valid_destinations = set(uid), set(did)
    issues: dict[str, object] = {}
    table_stats = {}
    for name, rows in datasets.items():
        key = {"users": "UserID", "destinations": "DestinationID", "reviews": "ReviewID", "history": "HistoryID"}[name]
        keys = [row.get(key, "") for row in rows]
        table_stats[name] = {
            "rows": len(rows),
            "duplicate_primary_key_count": len(keys) - len(set(keys)),
            "empty_by_column": {col: sum(not row.get(col, "") for row in rows) for col in REQUIRED[name]},
            "exact_duplicate_rows": len(rows) - len({tuple(sorted(row.items())) for row in rows}),
        }

    pairs = defaultdict(list)
    orphan_rows = Counter()
    invalid_ratings = Counter()
    for name, rating_col in (("reviews", "Rating"), ("history", "ExperienceRating")):
        for row in datasets[name]:
            u, d = row["UserID"], row["DestinationID"]
            if u not in valid_users or d not in valid_destinations:
                orphan_rows[name] += 1
                continue
            value = _rating(row[rating_col])
            if value is None:
                invalid_ratings[name] += 1
                continue
            pairs[(u, d)].append((name, value))

    by_user = Counter(u for (u, _d) in pairs)
    counts = Counter(by_user.values())
    ratings = {name: dict(sorted(Counter(row[col] for row in datasets[name]).items())) for name, col in (("reviews", "Rating"), ("history", "ExperienceRating"))}
    dates = [row["VisitDate"] for row in datasets["history"]]
    valid_dates = sorted(d for d in dates if d and _valid_date(d))
    conflicting = sum(1 for events in pairs.values() if any(r >= 4 for _, r in events) and any(r < 4 for _, r in events))
    pair_sources = Counter(tuple(sorted({source for source, _ in events})) for events in pairs.values())
    preferences = Counter(row["Preferences"] for row in users if row["Preferences"])
    types = Counter(row["Type"] for row in destinations if row["Type"])

    issues["orphan_rows_by_source"] = dict(orphan_rows)
    issues["invalid_ratings_by_source"] = dict(invalid_ratings)
    issues["conflicting_label_pairs"] = conflicting
    result = {
        "note": "Datos reales/sintéticos: aún NO verificado con la documentación de Kaggle.",
        "tables": table_stats,
        "users_unique": len(valid_users),
        "destinations_unique": len(valid_destinations),
        "pairs_unique_valid": len(pairs),
        "density": len(pairs) / (len(valid_users) * len(valid_destinations)) if valid_users and valid_destinations else None,
        "activity_users": {
            "0": len(valid_users) - len(by_user),
            "1": counts[1],
            "2": counts[2],
            "3_or_more": sum(n for count, n in counts.items() if count >= 3),
        },
        "rating_distribution": ratings,
        "destination_types": dict(sorted(types.items())),
        "preferences_distinct_count": len(preferences),
        "preferences_examples": [p for p, _ in preferences.most_common(10)],
        "history_date_valid_count": len(valid_dates),
        "history_date_invalid_count": sum(bool(d) and not _valid_date(d) for d in dates),
        "history_date_missing_count": sum(not d for d in dates),
        "history_date_min": min(valid_dates) if valid_dates else None,
        "history_date_max": max(valid_dates) if valid_dates else None,
        "reviews_have_no_timestamp": True,
        "pair_source_count": {"+".join(key): count for key, count in pair_sources.items()},
        "positive_threshold_proposal": 4,
        "issues": issues,
        "leakage_warning": "Popularity del CSV podría ser información agregada de origen desconocido: NO usar como predictor antes de verificar cómo se construyó.",
    }
    return result


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_processed(datasets: dict[str, list[dict[str, str]]], target: Path) -> dict:
    """Solo copia columnas suficientes para modelos; sin nombres, emails ni reseñas."""
    catalog = [
        {"destination_id": r["DestinationID"], "destination_name": r["Name"], "state": r["State"], "type": r["Type"], "best_time_to_visit": r["BestTimeToVisit"]}
        for r in datasets["destinations"]
    ]
    profiles = [
        {"user_id": r["UserID"], "preferences": r["Preferences"], "number_of_adults": r["NumberOfAdults"], "number_of_children": r["NumberOfChildren"]}
        for r in datasets["users"]
    ]
    observations = []
    user_ids = {r["user_id"] for r in profiles}
    dest_ids = {r["destination_id"] for r in catalog}
    dropped = Counter()
    for src, col in (("reviews", "Rating"), ("history", "ExperienceRating")):
        for row in datasets[src]:
            u, d = row["UserID"], row["DestinationID"]
            rating = _rating(row[col])
            if not u or not d or u not in user_ids or d not in dest_ids or rating is None:
                dropped[src] += 1
                continue
            observations.append({"user_id": u, "destination_id": d, "source": src, "rating": rating, "visit_date": row["VisitDate"] if src == "history" else ""})

    catalog.sort(key=lambda r: r["destination_id"])
    profiles.sort(key=lambda r: r["user_id"])
    observations.sort(key=lambda r: (r["user_id"], r["destination_id"], r["source"], r["visit_date"]))
    write_csv(target / "catalog.csv", ["destination_id", "destination_name", "state", "type", "best_time_to_visit"], catalog)
    write_csv(target / "profiles.csv", ["user_id", "preferences", "number_of_adults", "number_of_children"], profiles)
    write_csv(target / "interactions.csv", ["user_id", "destination_id", "source", "rating", "visit_date"], observations)
    return {"catalog_rows": len(catalog), "profiles_rows": len(profiles), "interaction_events_rows": len(observations), "dropped_invalid_observations": dict(dropped)}
