"""Entrenamiento offline dos torres: búsqueda, 30 semillas, prueba final sellada.

La validación selecciona hiperparámetros y checkpoint; nunca se usa test
para seleccionar arquitectura o pesos. Desconocidos NO son negativos reales.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from .data import write_json
from .evaluation import load_context, rank_ids
from .metrics import rank_metrics
from .neural import (DEFAULT_NEURAL, arrays_for_pairs, build_bundle, create_model,
                     load_checkpoint, save_checkpoint, score_catalog)

DEFAULT_CONFIG = {"embedding_dim": 8, "hidden_dim": 32, "latent_dim": 16,
                  "dropout": 0.1, "learning_rate": 0.005, "weight_decay": 0.0001,
                  "batch_size": 128, "negative_ratio": 2.0, "patience": 4}


def evaluate_model(model, bundle: dict, context: dict, split: str, k: int) -> dict:
    if split not in context["truth"]:
        raise ValueError(f"Split desconocido: {split}")
    all_values, union = [], set()
    few, many = [], []
    examples = []
    for user, relevant in sorted(context["truth"][split].items()):
        allowed = context["catalog_ids"] - context["observed"].get(user, set())
        scores = score_catalog(model, bundle, user)
        ranked = rank_ids(scores, allowed, k)
        union.update(ranked)
        values = rank_metrics(ranked, relevant, k)
        all_values.append(values)
        bucket = few if len(context["train"].get(user, set())) < 2 else many
        bucket.append(values)
        if len(examples) < 5:
            examples.append({"user_id": user, "recommended": ranked,
                             "heldout_count": len(relevant)})
    mean = lambda values, field: sum(v[field] for v in values) / len(values) if values else None
    return {"split": split, "users_evaluated": len(all_values),
            "precision_at_k": mean(all_values, "precision"),
            "recall_at_k": mean(all_values, "recall"),
            "ndcg_at_k": mean(all_values, "ndcg"),
            "catalog_coverage": len(union) / len(context["catalog_ids"]) if context["catalog_ids"] else None,
            "segments": {"few": {"users": len(few), "ndcg_at_k": mean(few, "ndcg")},
                         "many": {"users": len(many), "ndcg_at_k": mean(many, "ndcg")}},
            "examples": examples}


def fit_one(bundle: dict, context: dict, config: dict, *, seed: int, epochs: int, k: int = 5):
    if epochs < 1 or int(config["batch_size"]) < 1 or float(config["negative_ratio"]) < 0:
        raise ValueError("epochs y batch_size deben ser positivos; negative_ratio no negativo")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = create_model(bundle, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss()
    # BCE de validacion: pares observados reservados y nunca usados para
    # optimizar. No se inventan negativos a partir de destinos no visitados.
    val_examples = [
        (r["user_id"], r["destination_id"], float(r["split"] == "validation"))
        for r in context["manifest"]
        if r["split"] in ("validation", "validation_nonpositive")
    ]
    val_positive = sum(y == 1.0 for _u, _d, y in val_examples)
    val_negative = len(val_examples) - val_positive
    if not val_positive or not val_negative:
        raise ValueError("BCE de validacion requiere positivos y no positivos separados; "
                         "regenera prepare-entities en otra carpeta")
    positives = [(u, d) for u, d, y in bundle["examples"] if y == 1]
    negatives = [(u, d) for u, d, y in bundle["examples"] if y == 0]
    max_neg = min(len(negatives), round(len(positives) * float(config["negative_ratio"])))
    if max_neg < 1:
        raise ValueError("Se necesita al menos una valoración no positiva para entrenar")
    best_metric = -1.0
    best_weights = None
    best_epoch = 0
    best_loss = math.inf
    stale = 0
    curves = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # Submuestreo SOLO de calificaciones explícitas <= 3, no de ausencias.
        neg_epoch = rng.sample(negatives, k=max_neg)
        examples = [(u, d, 1.0) for u, d in positives] + [(u, d, 0.0) for u, d in neg_epoch]
        rng.shuffle(examples)
        model.train()
        total_loss = 0.0
        for start in range(0, len(examples), int(config["batch_size"])):
            batch = examples[start:start + int(config["batch_size"])]
            input_tensors = arrays_for_pairs(bundle, [(u, d) for u, d, _ in batch])
            labels = torch.tensor([y for _, _, y in batch], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*input_tensors)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        loss_train = total_loss / len(examples)
        model.eval()
        validation_loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(val_examples), int(config["batch_size"])):
                batch = val_examples[start:start + int(config["batch_size"])]
                val_inputs = arrays_for_pairs(bundle, [(u, d) for u, d, _y in batch])
                val_labels = torch.tensor([y for _u, _d, y in batch], dtype=torch.float32)
                val_logits = model(*val_inputs)
                validation_loss_sum += loss_fn(val_logits, val_labels).item() * len(batch)
        loss_validation = validation_loss_sum / len(val_examples)
        val_metrics = evaluate_model(model, bundle, context, "validation", k)
        score = val_metrics["ndcg_at_k"]
        # Con validación sin usuarios/positivos, hay que revisar split antes de entrenar.
        if score is None:
            raise ValueError("Validación vacía: no se puede elegir checkpoint")
        curves.append({"epoch": epoch, "train_bce": round(loss_train, 7),
                       "validation_bce": round(loss_validation, 7),
                       "validation_ndcg_at_k": score})
        if score > best_metric + 1e-12:
            best_metric, best_loss = score, loss_train
            best_weights, best_epoch, stale = deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break
    model.load_state_dict(best_weights)
    model.eval()
    result = evaluate_model(model, bundle, context, "validation", k)
    result.update({"seed": seed, "best_epoch": best_epoch, "epochs_ran": len(curves),
                   "train_loss_at_best": best_loss,
                   "validation_bce_at_best": next(p["validation_bce"] for p in curves
                                                  if p["epoch"] == best_epoch),
                   "validation_bce_sample_size": len(val_examples),
                   "validation_bce_positive_count": val_positive,
                   "validation_bce_nonpositive_count": val_negative,
                   "seconds": round(time.perf_counter() - started, 3),
                   "parameters": model.count_parameters(), "curves": curves,
                   "explicit_positive_count": len(positives), "explicit_nonpositive_pool": len(negatives),
                   "sampled_nonpositives_per_epoch": max_neg})
    return model, result


def _summary(reports: list[dict], keys: tuple[str, ...]) -> dict:
    result = {"seeds": len(reports), "per_metric": {}}
    for key in keys:
        values = [float(r[key]) for r in reports if r.get(key) is not None]
        result["per_metric"][key] = {
            "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "best": max(values) if values else None, "worst": min(values) if values else None,
        }
    return result


def tune(processed: Path, output: Path, *, seed: int = 42, epochs: int = 12,
         trials: int = 3, k: int = 5) -> dict:
    if trials < 1 or trials > 3:
        raise ValueError("trials debe estar entre 1 y 3 para el presupuesto predefinido")
    bundle, context = build_bundle(processed), load_context(processed)
    configs = [
        {**DEFAULT_CONFIG, "embedding_dim": 8, "hidden_dim": 16, "latent_dim": 16},
        {**DEFAULT_CONFIG, "embedding_dim": 16, "hidden_dim": 32, "latent_dim": 24, "dropout": 0.2},
        {**DEFAULT_CONFIG, "embedding_dim": 8, "hidden_dim": 32, "latent_dim": 16, "learning_rate": 0.001},
    ]
    attempts = []
    best = None
    for i, config in enumerate(configs[:trials]):
        _, metrics = fit_one(bundle, context, config, seed=seed, epochs=epochs, k=k)
        attempts.append({"trial": i + 1, "config": config, "validation": metrics})
        if best is None or (metrics["ndcg_at_k"], -metrics["train_loss_at_best"]) > (
                best["validation"]["ndcg_at_k"], -best["validation"]["train_loss_at_best"]):
            best = attempts[-1]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "search_results.json", {"selection_split": "validation", "k": k,
                                                       "trials": attempts})
    write_json(output / "best_config.json", best["config"])
    return {"trials": len(attempts), "best_trial": best["trial"], "best_validation_ndcg": best["validation"]["ndcg_at_k"],
            "best_config_file": str(output / "best_config.json"),
            "note": "Sólo validación; test y cold_test no se consultan en la búsqueda."}


def train_seeds(processed: Path, output: Path, artifacts: Path, *, seeds: int = 30,
                seed_start: int = 42, epochs: int = 16, k: int = 5) -> dict:
    if not 1 <= seeds <= 300:
        raise ValueError("seeds debe ser de 1 a 300")
    config_path = output / "best_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else DEFAULT_CONFIG
    bundle, context = build_bundle(processed), load_context(processed)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports = []
    best = None
    for seed in range(seed_start, seed_start + seeds):
        model, metrics = fit_one(bundle, context, config, seed=seed, epochs=epochs, k=k)
        save_checkpoint(artifacts / f"seed_{seed}.pt", model, bundle, config, seed, metrics["ndcg_at_k"])
        reports.append(metrics)
        if best is None or (metrics["ndcg_at_k"], -metrics["train_loss_at_best"]) > (
                best["ndcg_at_k"], -best["train_loss_at_best"]):
            best = metrics
            save_checkpoint(artifacts / "best.pt", model, bundle, config, seed, metrics["ndcg_at_k"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {"selection_split": "validation", "k": k, "config": config,
               "selected_seed": best["seed"], "model_parameters": best["parameters"],
               "metrics": _summary(reports, ("precision_at_k", "recall_at_k", "ndcg_at_k", "catalog_coverage")),
               "total_train_seconds": round(sum(r["seconds"] for r in reports), 3),
               "uses_only_explicit_ratings_as_negatives": True,
               "validation_bce_uses_heldout_explicit_nonpositives": True,
               "test_evaluated": False}
    write_json(output / "neural_validation_runs.json", {"runs": reports, "summary": summary})
    write_json(output / "neural_validation_summary.json", summary)
    return summary


def evaluate_saved(processed: Path, output: Path, artifacts: Path, *, split: str,
                   k: int = 5) -> dict:
    if split not in ("validation", "test", "cold_test"):
        raise ValueError("split inválido")
    files = sorted(artifacts.glob("seed_*.pt"))
    if not files:
        raise FileNotFoundError("Faltan checkpoints seed_*.pt; ejecuta train-neural")
    context = load_context(processed)
    results = []
    for path in files:
        model, bundle, saved = load_checkpoint(path, processed)
        row = evaluate_model(model, bundle, context, split, k)
        row["seed"] = saved["seed"]
        results.append(row)
    best_model, best_bundle, selected = load_checkpoint(artifacts / "best.pt", processed)
    selected_metrics = evaluate_model(best_model, best_bundle, context, split, k)
    summary = {"split": split, "k": k, "seed_count": len(results),
               "metrics": _summary(results, ("precision_at_k", "recall_at_k", "ndcg_at_k", "catalog_coverage")),
               "selected_by_validation_seed": selected["seed"],
               "selected_model_metrics": selected_metrics,
               "warning": "No seleccionar hiperparámetros ni semillas a partir de test/cold_test."}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"neural_{split}_results.json", {"results": results, "summary": summary})
    return summary
