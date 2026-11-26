"""CLI P4. Todos los experimentos actuales funcionan sin paquetes externos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import evaluate_popularity
from .classical import train_item_knn
from .data import audit, export_processed, load_all, write_json
from .evaluation import compare_models
from .infer import recommend
from .splits import build_split


def run() -> None:
    parser = argparse.ArgumentParser(description="P4 - Recomendador de destinos")
    parser.add_argument("command", choices=["audit", "prepare", "baseline", "fit-classical", "compare", "recommend", "tune-neural", "train-neural", "eval-neural", "audit-catalog", "report-p4", "figures-p4", "campaign-p4", "audit-entities", "prepare-entities"])
    parser.add_argument("--entity-output", type=Path, default=Path("data/processed/recommender_entities"))
    parser.add_argument("--report-dir", type=Path, default=Path("experiments/m3/diagnostics"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--audience-type", default="beach")
    parser.add_argument("--exposure-rate", type=float, default=0.50)
    parser.add_argument("--conversion-rate", type=float, default=0.03)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--neural-artifacts", type=Path, default=Path("artifacts/recommender/neural"))
    parser.add_argument("--results", type=Path, default=Path("experiments/m3/neural"))
    parser.add_argument("--source", type=Path, default=Path("../DATASET/archive"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/recommender"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/recommender/item_knn.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cold-fraction", type=float, default=0.15)
    parser.add_argument("--positive-at", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--split", choices=["validation", "test", "cold_test"], default="validation")
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--preferences", default=None)
    parser.add_argument("--preference-boost", type=float, default=0.02)
    parser.add_argument("--method", choices=["popularity", "item-knn", "content", "neural", "neural-hybrid"], default="content")
    args = parser.parse_args()

    if args.command in ("audit", "prepare"):
        datasets = load_all(args.source)
        report = audit(datasets)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.command == "prepare":
            write_json(args.output / "audit.json", report)
            print("PROCESADOS:", json.dumps(export_processed(datasets, args.output), ensure_ascii=False))
            print("PARTICIÓN:", json.dumps(build_split(args.output, seed=args.seed, cold_fraction=args.cold_fraction,
                                                         positive_at=args.positive_at), ensure_ascii=False, indent=2))
    elif args.command == "baseline":
        result = evaluate_popularity(args.output, args.k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "fit-classical":
        result = train_item_knn(args.output, args.artifact, neighbors=args.neighbors)
        print(json.dumps({key: value for key, value in result.items() if key not in ("neighbors_by_item", "positive_counts", "catalog_ids")},
                         ensure_ascii=False, indent=2))
        print("Modelo serializado:", args.artifact.resolve())
    elif args.command == "compare":
        result = compare_models(args.output, args.artifact, k=args.k, split=args.split)
        report_file = Path("experiments/m3") / f"comparison_{args.split}_k{args.k}.json"
        write_json(report_file, result)
        print(json.dumps({**result, "models": {key: {k: v for k, v in model.items() if k != "example_rankings"}
                                                      for key, model in result["models"].items()}}, ensure_ascii=False, indent=2))
        print("Reporte completo:", report_file.resolve())
    elif args.command in ("tune-neural", "train-neural", "eval-neural"):
        # Torch se importa sólo cuando se necesita, para que baselines y tests
        # existentes continúen funcionando sin esta dependencia.
        from .train import evaluate_saved, train_seeds, tune
        if args.command == "tune-neural":
            result = tune(args.output, args.results, seed=args.seed, epochs=args.epochs,
                          trials=args.trials, k=args.k)
        elif args.command == "train-neural":
            result = train_seeds(args.output, args.results, args.neural_artifacts,
                                 seeds=args.seeds, seed_start=args.seed_start,
                                 epochs=args.epochs, k=args.k)
        else:
            result = evaluate_saved(args.output, args.results, args.neural_artifacts,
                                    split=args.split, k=args.k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "audit-entities":
        from .entities import report_semantic_overlap
        result = report_semantic_overlap(args.output, args.report_dir / "m3_entity_leakage_diagnostic.json")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "prepare-entities":
        from .entities import prepare_entities
        result = prepare_entities(args.output, args.entity_output, positive_at=args.positive_at)
        split = build_split(args.entity_output, seed=args.seed, cold_fraction=args.cold_fraction,
                            positive_at=args.positive_at)
        print(json.dumps({"entity_dataset": result, "new_split": split}, ensure_ascii=False, indent=2))
    elif args.command == "audit-catalog":
        from .audits import catalog_audit
        output = args.report_dir / "m3_catalog_audit.json"
        result = catalog_audit(args.output, output)
        print(json.dumps({key: (value if key not in ("same_name", "same_name_and_state", "same_name_state_and_type")
                                else {k: v for k, v in value.items() if k != "examples"}) for key, value in result.items()},
                         ensure_ascii=False, indent=2))
        print("Detalle de nombres/IDs repetidos:", output)
    elif args.command == "report-p4":
        from .reporting import generate_report
        result = generate_report(args.output, args.artifact, args.neural_artifacts / "best.pt", args.report_dir,
                                 split=args.split, ks=tuple(args.ks))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "figures-p4":
        from .plots import make_figures
        result = make_figures(args.output, args.report_dir, args.neural_artifacts / "best.pt",
                              args.figure_dir, split=args.split,
                              training_results=args.results)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "campaign-p4":
        from .business import campaign_scenario
        path = args.report_dir / "m3_campaign_scenario.json"
        result = campaign_scenario(args.output, path, destination_type=args.audience_type,
                                   exposure_rate=args.exposure_rate, assumed_conversion_rate=args.conversion_rate)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("Escenario guardado:", path)
    else:
        profile = {"preferences": args.preferences} if args.preferences is not None else None
        result = recommend(user_id=args.user_id, profile=profile, k=args.k, method=args.method,
                           processed=args.output, artifact=args.artifact,
                           neural_artifact=args.neural_artifacts / "best.pt",
                            preference_boost=args.preference_boost)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
