"""Capa de datos, puntuación y artefactos de red neuronal P4.

Las únicas etiquetas de entrenamiento son train=1 y observed_nonpositive=0.
NO se muestrean ausencias como falsos negativos. El ratio de negativos indica
submuestreo de calificaciones explícitas 1-3, no de destinos desconocidos.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from .coldstart import parse_preferences
from .data import read_csv
from .models import TwoTowerRecommender

TYPES = ("adventure", "beach", "city", "historical", "nature")
DEFAULT_NEURAL = Path("artifacts/recommender/neural/best.pt")


def build_bundle(processed: Path) -> dict:
    catalog, _ = read_csv(processed / "catalog.csv")
    profiles, _ = read_csv(processed / "profiles.csv")
    manifest, _ = read_csv(processed / "split_manifest.csv")
    item_by_id = {r["destination_id"]: r for r in catalog}
    profile_by_id = {r["user_id"]: r for r in profiles}
    if len(item_by_id) != len(catalog) or len(profile_by_id) != len(profiles):
        raise ValueError("Hay identificadores duplicados")
    for item in catalog:
        if item["type"].lower() not in TYPES:
            raise ValueError(f"Categoría no reconocida: {item['type']}")

    # Ambos vocabularios dependen SOLO de registros que se pueden entrenar.
    fit_rows = [r for r in manifest if r["split"] in ("train", "observed_nonpositive")]
    users = sorted({r["user_id"] for r in fit_rows})
    items = sorted({r["destination_id"] for r in fit_rows})
    user_index = {v: i + 1 for i, v in enumerate(users)}
    item_index = {v: i + 1 for i, v in enumerate(items)}
    examples = [(r["user_id"], r["destination_id"], int(r["split"] == "train")) for r in fit_rows]
    if not any(label for _, _, label in examples) or not any(not label for _, _, label in examples):
        raise ValueError("Se requieren tanto calificaciones positivas como no positivas para BCE")
    return {"catalog": catalog, "profiles": profile_by_id, "items": item_by_id,
            "manifest": manifest, "user_index": user_index, "item_index": item_index,
            "examples": examples}


def _multi_hot(values) -> list[float]:
    return [1.0 if typ in values else 0.0 for typ in TYPES]


def arrays_for_pairs(bundle: dict, pairs: list[tuple[str | None, str]],
                     profile: dict | None = None):
    uids, iids, prefs, types = [], [], [], []
    for user, item in pairs:
        row = bundle["items"].get(item)
        if row is None:
            raise ValueError(f"Destino desconocido: {item}")
        candidate_profile = profile if profile is not None else bundle["profiles"].get(user, {})
        preferences = candidate_profile.get("preferences", candidate_profile.get("Preferences", ""))
        uids.append(bundle["user_index"].get(user, 0))
        iids.append(bundle["item_index"].get(item, 0))
        prefs.append(_multi_hot(parse_preferences(preferences)))
        types.append(_multi_hot({row["type"].lower()}))
    return (
        torch.tensor(uids, dtype=torch.long),
        torch.tensor(iids, dtype=torch.long),
        torch.tensor(prefs, dtype=torch.float32).reshape(-1, len(TYPES)),
        torch.tensor(types, dtype=torch.float32).reshape(-1, len(TYPES)),
    )


def create_model(bundle: dict, config: dict) -> TwoTowerRecommender:
    return TwoTowerRecommender(n_users=len(bundle["user_index"]), n_items=len(bundle["item_index"]),
                               n_types=len(TYPES), embedding_dim=int(config["embedding_dim"]),
                               hidden_dim=int(config["hidden_dim"]), latent_dim=int(config["latent_dim"]),
                               dropout=float(config["dropout"]))


def save_checkpoint(path: Path, model: TwoTowerRecommender, bundle: dict, config: dict,
                    seed: int, val_score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config,
                "seed": seed, "val_ndcg_at_5": val_score,
                "user_index": bundle["user_index"], "item_index": bundle["item_index"],
                "types": list(TYPES)}, path)


def load_checkpoint(path: Path, processed: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}. Ejecuta train-neural antes de inferir")
    # pesos generados por nuestro proyecto, no cargar checkpoints externos.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(ckpt["types"]) != TYPES:
        raise ValueError("Tipos de contenido incompatibles")
    bundle = build_bundle(processed)
    if bundle["user_index"] != ckpt["user_index"] or bundle["item_index"] != ckpt["item_index"]:
        raise ValueError("Datos procesados no coinciden con el checkpoint; no reentrenar en inferencia")
    model = create_model(bundle, ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, bundle, ckpt


def score_catalog(model, bundle: dict, user_id: str | None, profile: dict | None = None) -> dict[str, float]:
    catalog_ids = sorted(bundle["items"])
    pairs = [(user_id, destination_id) for destination_id in catalog_ids]
    vectors = arrays_for_pairs(bundle, pairs, profile=profile)
    with torch.no_grad():
        logits = model(*vectors)
        probabilities = torch.sigmoid(logits).tolist()
    return dict(zip(catalog_ids, (float(v) for v in probabilities)))
