"""Gráficos P4 regenerables desde reportes, sin entrenamiento ni acceso a test.

Requiere matplotlib, numpy y torch. Guarda figuras PNG en reports/figures/.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def make_figures(processed: Path, reports: Path, neural_artifact: Path,
                 figures_dir: Path, *, split: str = "validation",
                 training_results: Path | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from .neural import arrays_for_pairs, load_checkpoint

    path = reports / f"m3_report_{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Primero ejecuta report-p4 --split {split}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    freqs = payload["training_frequency"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    created = []

    def save(fig, filename):
        path = figures_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

    # 1. Cola larga del entrenamiento, incluyendo usuarios/destinos con cero.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in zip(axes, ("positive_count_per_user", "positive_count_per_destination"),
                              ("Positivos por usuario", "Positivos por destino")):
        counts = Counter(freqs[key].values())
        ax.bar(sorted(counts), [counts[v] for v in sorted(counts)])
        ax.set(title=title, xlabel="Número de interacciones positivas (train)", ylabel="Cantidad")
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("M3 - Distribución de interacciones positivas (entrenamiento)")
    fig.tight_layout()
    save(fig, "m3_01_cola_larga_train.png")

    # 2. Tres métricas de ranking para K por modelo.
    methods = sorted({row["model"] for row in metrics})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    for ax, metric, title in zip(axes,
                                  ("precision_at_k", "recall_at_k", "ndcg_at_k"),
                                  ("Precision@K", "Recall@K", "NDCG@K")):
        for method in methods:
            rows = sorted((r for r in metrics if r["model"] == method), key=lambda r: r["k"])
            ax.plot([r["k"] for r in rows], [r[metric] for r in rows], "o-", label=method)
        ax.set(title=title, xlabel="K")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Promedio de usuarios")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"M3 - Comparación de ranking ({split})")
    fig.tight_layout()
    save(fig, f"m3_02_metricas_por_k_{split}.png")

    # 3. Cobertura y concentración frente a K.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for method in methods:
        rows = sorted((r for r in metrics if r["model"] == method), key=lambda r: r["k"])
        xs = [r["k"] for r in rows]
        axes[0].plot(xs, [r["catalog_coverage"] for r in rows], "o-", label=method)
        axes[1].plot(xs, [r["hhi"] for r in rows], "o-", label=method)
    axes[0].set(title="Cobertura del catálogo (por ID)", xlabel="K", ylabel="Fracción de ítems")
    axes[1].set(title="Concentración HHI (menor = más diverso)", xlabel="K", ylabel="HHI")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"M3 - Cobertura y concentración ({split})")
    fig.tight_layout()
    save(fig, f"m3_03_cobertura_concentracion_{split}.png")

    # 4. Embeddings finales DE LOS ÍTEMS VISTOS EN TRAIN, con PCA en vez de t-SNE.
    # Nota: la consigna menciona t-SNE/UMAP como ejemplo; PCA es más simple y reproducible.
    model, bundle, ckpt = load_checkpoint(neural_artifact, processed)
    trained = list(sorted(bundle["item_index"], key=bundle["item_index"].get))
    if len(trained) >= 3:
        with torch.no_grad():
            inputs = arrays_for_pairs(bundle, [(None, d) for d in trained])
            dest_tensors = torch.cat([model.item_embedding(inputs[1]), inputs[3]], dim=-1)
            vectors = model.item_tower(dest_tensors).detach().cpu().numpy()
        vectors -= vectors.mean(axis=0, keepdims=True)
        # PCA por SVD, sin reentrenar redes ni utilizar las interacciones retenidas.
        _u, _s, vt = np.linalg.svd(vectors, full_matrices=False)
        xy = vectors @ vt[:2].T
        catalog = bundle["items"]
        fig, ax = plt.subplots(figsize=(8.5, 6))
        for typ in sorted({catalog[d]["type"] for d in trained}):
            idx = [i for i, d in enumerate(trained) if catalog[d]["type"] == typ]
            ax.scatter(xy[idx, 0], xy[idx, 1], s=12, alpha=0.55, label=typ)
        ax.set(xlabel="Componente principal 1", ylabel="Componente principal 2",
               title="Destinos presentes en entrenamiento - representación neuronal")
        ax.legend(title="Tipo", fontsize=8)
        ax.grid(alpha=0.15)
        fig.tight_layout()
        save(fig, "m3_04_embeddings_pca_train.png")

    # 5. Curvas de BCE train / validacion y NDCG de validacion.
    # Validacion BCE usa negativos explicitos reservados antes de entrenar.
    training_results = training_results or (reports.parent / "neural")
    run_file = training_results / "neural_validation_runs.json"
    if run_file.is_file():
        runs_payload = json.loads(run_file.read_text(encoding="utf-8"))
        runs = runs_payload["runs"]
        run_k = int(runs_payload.get("summary", {}).get("k", 5))
        if run_k not in {int(m["k"]) for m in metrics}:
            raise ValueError(f"Curvas de entrenamiento NDCG@{run_k} incompatibles con K del reporte")
        if int(ckpt["seed"]) not in {int(r["seed"]) for r in runs}:
            raise ValueError("El checkpoint seleccionado no pertenece a las semillas del reporte de entrenamiento")
        if runs:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for row in runs:
                curve = row["curves"]
                ax.plot([p["epoch"] for p in curve], [p["train_bce"] for p in curve],
                        alpha=0.14, linewidth=0.8)
            for row in runs:
                curve = row["curves"]
                point = next((p for p in curve if p["epoch"] == row["best_epoch"]), None)
                if point:
                    ax.scatter(point["epoch"], point["train_bce"], s=7, marker="x", alpha=0.35)
            ax.set(xlabel="Época", ylabel="BCE de entrenamiento",
                   title="M3 - Curvas BCE de entrenamiento por semilla")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            save(fig, "m3_05_curvas_bce_train.png")
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for row in runs:
                curve = row["curves"]
                ax.plot([p["epoch"] for p in curve], [p["validation_ndcg_at_k"] for p in curve],
                        alpha=0.25, linewidth=0.8)
            ax.set(xlabel="Época", ylabel="NDCG de validación",
                   title="M3 - NDCG de validación por época y semilla")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            save(fig, "m3_06_curvas_ndcg_val.png")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ndcg = [r["ndcg_at_k"] for r in runs]
            ax.boxplot([ndcg], tick_labels=[f"Dos torres ({len(ndcg)} semillas)"])
            ax.set(ylabel=f"NDCG@{run_k} validación", title=f"M3 - Variabilidad NDCG@{run_k} entre semillas")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            save(fig, "m3_07_boxplot_semillas.png")
            selected_run = next((r for r in runs if int(r["seed"]) == int(ckpt["seed"])), None)
            if selected_run and selected_run["curves"] and all(
                    "validation_bce" in point for point in selected_run["curves"]):
                fig, ax = plt.subplots(figsize=(9, 4.5))
                curve = selected_run["curves"]
                xs = [p["epoch"] for p in curve]
                ax.plot(xs, [p["train_bce"] for p in curve], "o-", label="BCE entrenamiento")
                ax.plot(xs, [p["validation_bce"] for p in curve], "o-", label="BCE validación")
                ax.axvline(int(selected_run["best_epoch"]), color="gray", linestyle="--",
                           linewidth=1, label="Mejor época (NDCG)")
                ax.set(xlabel="Época", ylabel="BCEWithLogitsLoss promedio",
                       title=f"M3 - Pérdida train vs validación (semilla {ckpt['seed']})")
                ax.legend()
                ax.grid(alpha=0.2)
                fig.tight_layout()
                save(fig, "m3_10_curvas_bce_train_validation.png")

    search_file = training_results / "search_results.json"
    if search_file.is_file():
        search_payload = json.loads(search_file.read_text(encoding="utf-8"))
        trial_k = int(search_payload.get("k", 5))
        if trial_k not in {int(m["k"]) for m in metrics}:
            raise ValueError(f"Hiperparámetros NDCG@{trial_k} incompatibles con K del reporte")
        trials = search_payload["trials"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([f"Trial {t['trial']}" for t in trials], [t["validation"]["ndcg_at_k"] for t in trials])
        ax.set(title="M3 - Búsqueda de hiperparámetros", ylabel=f"NDCG@{trial_k} en validación")
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        save(fig, "m3_08_busqueda_hiperparametros.png")

    # 6. Diagrama explicativo de la arquitectura final con dimensiones reales.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    dim_embedding = model.user_embedding.embedding_dim
    dim_in = model.user_tower[0].in_features
    dim_hidden = model.user_tower[0].out_features
    dim_latent = model.user_tower[-1].out_features
    n_params = model.count_parameters()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    def node(x, y, w, h, label, face):
        shape = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.14", linewidth=1,
                               edgecolor="#334155", facecolor=face)
        ax.add_patch(shape)
        ax.text(x + w/2, y+h/2, label, ha="center", va="center", fontsize=9)
    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.25, color="#475569"))
    node(0.45, 3.9, 2.2, 1.0, f"Usuario\nEmbedding ID: {dim_embedding}\nPreferences: 5", "#dbeafe")
    node(0.45, 1.0, 2.2, 1.0, f"Destino\nEmbedding ID: {dim_embedding}\nType: 5", "#d1fae5")
    node(4.0, 3.9, 2.45, 1.0, f"Torre de usuario\n{dim_in} → {dim_hidden} → {dim_latent}", "#bfdbfe")
    node(4.0, 1.0, 2.45, 1.0, f"Torre de destino\n{dim_in} → {dim_hidden} → {dim_latent}", "#bbf7d0")
    node(7.9, 2.4, 1.6, 1.0, f"Producto\ninterno\n/ sqrt({dim_latent})", "#fde68a")
    node(10.35, 2.4, 1.25, 1.0, "Sigmoid\nscore", "#fee2e2")
    arrow((2.65, 4.4), (4, 4.4))
    arrow((2.65, 1.5), (4, 1.5))
    arrow((6.45, 4.4), (7.9, 3.15))
    arrow((6.45, 1.5), (7.9, 2.65))
    arrow((9.5, 2.9), (10.35, 2.9))
    ax.text(6, 5.5, f"Red de dos torres | Parámetros entrenables: {n_params:,}".replace(",", "."),
            ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(6, 0.45, "Loss: BCEWithLogits (train) | Clases: rating ≥4 positivo, rating 1–3 no positivo",
            ha="center", fontsize=9)
    save(fig, "m3_09_arquitectura_dos_torres.png")

    return {"figure_count": len(created), "figures": created,
            "training_results_source": str(training_results),
            "note": "PCA de ítems vistos en entrenamiento; no valida clusters causales. BCE(validation) exige una reserva de no-positivos explícitos."}
