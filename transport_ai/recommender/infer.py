"""Interfaz para P5: inferencia de referencias sin reentrenamiento.

Esta implementación inicial NO es el recomendador neuronal final.
"""
from __future__ import annotations

from pathlib import Path

from .classical import load_knn
from .coldstart import score_content
from .reranking import rerank_by_preferences
from .evaluation import rank_ids
from .classical import score_knn
from .data import read_csv
from collections import Counter, defaultdict


DEFAULT_PROCESSED = Path("data/processed/recommender")
DEFAULT_ARTIFACT = Path("artifacts/recommender/item_knn.json")


def recommend(user_id: str | None = None, profile: dict | None = None, k: int = 5,
              method: str = "content", processed: Path = DEFAULT_PROCESSED,
              artifact: Path = DEFAULT_ARTIFACT,
              neural_artifact: Path = Path("artifacts/recommender/neural/best.pt"),
               preference_boost: float = 0.02) -> list[dict]:
    """Devuelve lista ordenada sin repetir, con historial train/observado únicamente.

    `profile={'preferences': 'Beach, Nature'}` para un nuevo usuario.
    `method='item-knn'` para un usuario existente (requiere artefacto guardado).
    """
    if k < 1:
        raise ValueError("k debe ser >= 1")
    # Solo cargar entrenamiento e historial disponible. Nunca se construye
    # ground truth de test/validación dentro del servicio de inferencia.
    catalog_rows, _ = read_csv(processed / "catalog.csv")
    profiles_rows, _ = read_csv(processed / "profiles.csv")
    manifest, _ = read_csv(processed / "split_manifest.csv")
    catalog = {c["destination_id"]: c for c in catalog_rows}
    profiles = {p["user_id"]: p for p in profiles_rows}
    train = defaultdict(set)
    observed = defaultdict(set)
    popularity = Counter()
    for row in manifest:
        part = row["split"]
        if part in ("train", "observed_nonpositive"):
            observed[row["user_id"]].add(row["destination_id"])
        if part == "train":
            train[row["user_id"]].add(row["destination_id"])
            popularity[row["destination_id"]] += 1
    if user_id is not None and user_id not in profiles:
        if profile is None:
            raise ValueError("Usuario desconocido: suministra profile para un usuario nuevo")
        user_id = None
    if user_id is None and profile is None:
        raise ValueError("Debe indicar user_id o profile")
    if user_id is None and method == "item-knn":
        raise ValueError("item-knn necesita historial; para nuevos usuarios usa 'content' o 'popularity'")
    knn = load_knn(artifact) if method == "item-knn" else None
    user = user_id or ""
    if method in ("neural", "neural-hybrid"):
        from .neural import load_checkpoint, score_catalog
        model, bundle, _saved = load_checkpoint(neural_artifact, processed)
        selected_profile = profile if profile is not None else (profiles[user] if user_id else None)
        scores = score_catalog(model, bundle, user_id, profile=selected_profile)
        if method == "neural-hybrid":
            preferences = (selected_profile.get("preferences", selected_profile.get("Preferences", ""))
                           if selected_profile else "")
            scores = rerank_by_preferences(scores, catalog, preferences, preference_boost)
    elif method == "content":
        preferences = profile.get("preferences", "") if profile is not None else profiles[user]["preferences"]
        scores = score_content(catalog_rows, preferences, popularity)
    else:
        if method == "item-knn":
            scores = score_knn(knn, train.get(user, set()))
        elif method == "popularity":
            scores = dict(popularity)
        else:
            raise ValueError(f"Modelo desconocido: {method}")
    seen = observed.get(user, set()) if user_id is not None else set()
    allowed = set(catalog) - seen
    history_names = [catalog[d]["destination_name"] for d in sorted(train.get(user, set()))]
    ordered = rank_ids(scores, allowed, k)
    result = []
    for d in ordered:
        dest = catalog[d]
        if method == "content":
            reason = "Coincidencia con preferencias declaradas" if scores.get(d, 0) >= 1 else "Alternativa del catálogo"
        elif method == "neural-hybrid":
            reason = "Ranking neuronal con bonificacion explicita de preferencias (experimental)"
        elif method == "neural":
            reason = "Ranking neuronal: identificadores y categorías/preferencias"
        elif method == "item-knn":
            reason = "Similitud colaborativa o respaldo por popularidad (entrenamiento)"
        else:
            reason = "Popularidad calculada en entrenamiento"
        result.append({"destination_id": d, "destination_name": dest["destination_name"],
                       "score": float(scores.get(d, 0.0)), "reason": reason,
                       "history": history_names})
    return result
