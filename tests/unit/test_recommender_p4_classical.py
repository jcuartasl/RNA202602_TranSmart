"""Pruebas para el baseline colaborativo y el contrato de inferencia de P4."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from transport_ai.recommender.classical import train_item_knn, load_knn, score_knn
from transport_ai.recommender.coldstart import parse_preferences, score_content
from transport_ai.recommender.data import FILES, load_all, export_processed
from transport_ai.recommender.splits import build_split
from transport_ai.recommender.evaluation import compare_models, load_context
from transport_ai.recommender.infer import recommend


class ClassicalTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        self.source.mkdir()
        data = {
            FILES["users"]: ("UserID,Name,Email,Preferences,Gender,NumberOfAdults,NumberOfChildren", [
                f"u{i},User{i},u{i}@example.com,{('Beaches; Historical' if i % 2 else 'Nature, Adventure')},X,2,0"
                for i in range(1, 12)]),
            FILES["destinations"]: ("DestinationID,Name,State,Type,Popularity,BestTimeToVisit", [
                f"d{i},Destino{i},Estado{i},{['Beach','Historical','Nature','Adventure','City'][i % 5]},0,January"
                for i in range(1, 11)]),
            FILES["reviews"]: ("ReviewID,DestinationID,UserID,Rating,ReviewText", [
                f"r{i},d{(i % 10) + 1},u{(i % 11) + 1},5,ok" for i in range(1, 22)]),
            FILES["history"]: ("HistoryID,UserID,DestinationID,VisitDate,ExperienceRating", [
                f"h{i},d{(i % 10) + 1},u{(i % 11) + 1},2024-01-01,5" for i in range(1, 2)]),
        }
        # Corrige orden UserID / DestinationID para history.
        data[FILES["history"]] = ("HistoryID,UserID,DestinationID,VisitDate,ExperienceRating",
                                   ["h1,u1,d7,2024-01-01,5", "h2,u3,d9,2024-01-02,4"])
        for fname, (header, rows) in data.items():
            (self.source / fname).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        self.processed = self.root / "processed"
        self.artifact = self.root / "knn.json"
        export_processed(load_all(self.source), self.processed)
        build_split(self.processed, seed=42, cold_fraction=0.20)

    def test_preference_parsing(self):
        self.assertEqual(parse_preferences("Beaches, Historical"), {"beach", "historical"})
        self.assertEqual(parse_preferences("Nature, Adventure"), {"nature", "adventure"})
        sample = [{"destination_id": "a", "type": "Beach"}, {"destination_id": "b", "type": "City"}]
        ranked = score_content(sample, "Beaches", {"b": 100})
        self.assertGreater(ranked["a"], ranked["b"])

    def test_train_knn_and_leakage(self):
        model = train_item_knn(self.processed, self.artifact, neighbors=3)
        self.assertEqual(model["fitted_splits"], ["train"])
        self.assertEqual(load_knn(self.artifact)["neighbors"], 3)
        context = load_context(self.processed)
        self.assertEqual(sum(model["positive_counts"].values()), sum(len(s) for s in context["train"].values()))
        self.assertTrue(all(value >= 0 for value in score_knn(model, {"d1"}).values()))
        cold = set(context["truth"]["cold_test"])
        self.assertFalse(cold.intersection(context["train"]))

    def test_compare_and_infer(self):
        train_item_knn(self.processed, self.artifact)
        metrics = compare_models(self.processed, self.artifact, k=3, split="validation")
        self.assertEqual(set(metrics["models"]), {"popularity", "item-knn", "content"})
        ctx = load_context(self.processed)
        for model in metrics["models"].values():
            self.assertTrue(model["ndcg_at_k"] is None or 0 <= model["ndcg_at_k"] <= 1)
        for user in list(ctx["profiles"])[:1]:
            output = recommend(user_id=user, k=5, processed=self.processed, artifact=self.artifact)
            self.assertEqual(len(output), 5)
            self.assertEqual(len(set(row["destination_id"] for row in output)), 5)
            self.assertFalse(set(row["destination_id"] for row in output) & ctx["observed"].get(user, set()))
        output = recommend(profile={"preferences": "Beach"}, k=3, processed=self.processed, artifact=self.artifact)
        self.assertEqual(len(output), 3)
        self.assertTrue(all(row["history"] == [] for row in output))
        self.assertTrue(all("reason" in row for row in output))


if __name__ == "__main__":
    unittest.main()
