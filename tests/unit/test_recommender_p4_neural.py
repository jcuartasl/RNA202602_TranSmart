"""Pruebas integradas con fixture sintético (NO resultados del dataset de Kaggle)."""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transport_ai.recommender.data import FILES, export_processed, load_all
from transport_ai.recommender.splits import build_split


class NeuralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa
        except ImportError as err:
            raise unittest.SkipTest("PyTorch no instalado: ejecutar pip install torch") from err

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        raw = root / "original"
        raw.mkdir()
        names = ["Beach", "Historical", "Nature", "Adventure", "City"]
        destinations = [f"d{i},Destino {i},Estado,{names[i % 5]},99,Enero" for i in range(1, 11)]
        users = [f"u{i},Usuario {i},u{i}@test.com,{('Beaches; Historical' if i % 2 else 'Nature; Adventure')},X,1,0"
                 for i in range(1, 13)]
        # Tres positivos por usuario para obtener split warm de validación/test.
        reviews = [f"r{i*10+j},d{(i+j) % 10 + 1},u{i},5,ok" for i in range(1, 13) for j in range(3)]
        # Añade calificaciones negativas explícitas sin duplicar pares positivos.
        reviews += [f"rneg{i},d{(i+6) % 10 + 1},u{i},2,meh" for i in range(1, 13)]
        history = ["h1,u1,d10,2024-01-01,1"]
        table = {
            FILES["destinations"]: ("DestinationID,Name,State,Type,Popularity,BestTimeToVisit", destinations),
            FILES["users"]: ("UserID,Name,Email,Preferences,Gender,NumberOfAdults,NumberOfChildren", users),
            FILES["reviews"]: ("ReviewID,DestinationID,UserID,Rating,ReviewText", reviews),
            FILES["history"]: ("HistoryID,UserID,DestinationID,VisitDate,ExperienceRating", history),
        }
        for fname, (header, records) in table.items():
            (raw / fname).write_text("\n".join([header, *records]) + "\n", encoding="utf8")
        self.processed = root / "processed"
        self.results = root / "results"
        self.artifacts = root / "artifacts"
        export_processed(load_all(raw), self.processed)
        build_split(self.processed, seed=42, cold_fraction=0.10)

    def test_towers_train_and_inference_without_leakage(self):
        from transport_ai.recommender.neural import build_bundle, load_checkpoint, score_catalog
        from transport_ai.recommender.train import train_seeds, evaluate_saved
        from transport_ai.recommender.evaluation import load_context
        from transport_ai.recommender.infer import recommend

        bundle = build_bundle(self.processed)
        ctx = load_context(self.processed)
        training_users = set(bundle["user_index"])
        self.assertFalse(training_users & set(ctx["truth"]["cold_test"]))
        self.assertEqual(len([x for x in bundle["examples"] if x[2] == 1]),
                         sum(len(items) for items in ctx["train"].values()))
        report = train_seeds(self.processed, self.results, self.artifacts, seeds=2, epochs=3, k=3)
        self.assertEqual(report["metrics"]["per_metric"]["ndcg_at_k"]["mean"] is not None, True)
        self.assertEqual(report["test_evaluated"], False)
        self.assertTrue((self.artifacts / "best.pt").exists())
        model, bundle, metadata = load_checkpoint(self.artifacts / "best.pt", self.processed)
        self.assertTrue(model.count_parameters() > 0)
        scores = score_catalog(model, bundle, None, profile={"preferences": "Beach"})
        self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in scores.values()))
        ranked = recommend(profile={"preferences": "Beaches"}, k=3, method="neural",
                           processed=self.processed, neural_artifact=self.artifacts / "best.pt")
        self.assertEqual(len({row["destination_id"] for row in ranked}), 3)
        self.assertEqual(len(ranked), 3)
        self.assertTrue(all("score" in r and "reason" in r for r in ranked))
        result = evaluate_saved(self.processed, self.results, self.artifacts, split="validation", k=3)
        self.assertEqual(result["seed_count"], 2)


if __name__ == "__main__":
    unittest.main()
