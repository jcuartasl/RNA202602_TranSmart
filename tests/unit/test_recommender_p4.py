"""Tests autocontenidos P4. Ejecutar: python -m unittest discover -s tests/unit -p 'test_recommender_p4.py' -v"""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest

from transport_ai.recommender.baselines import evaluate_popularity
from transport_ai.recommender.data import FILES, audit, export_processed, load_all
from transport_ai.recommender.metrics import rank_metrics
from transport_ai.recommender.splits import build_split


class RecommenderTest(unittest.TestCase):
    def _fixture(self, root: Path):
        content = {
            FILES["users"]: ("UserID,Name,Email,Preferences,Gender,NumberOfAdults,NumberOfChildren", [
                "u1,Alex,a@example.org,Beach,X,2,0", "u2,B,B@example.org,Nature,X,1,0", "u3,C,c@example.org,City,X,1,0", "u4,D,d@example.org,Beach,X,1,0"]),
            FILES["destinations"]: ("DestinationID,Name,State,Type,Popularity,BestTimeToVisit", [
                "d1,Destino1,Uno,Beach,100,Summer", "d2,Destino2,Dos,City,90,All", "d3,Destino3,Tres,Nature,80,Winter", "d4,Destino4,Cuatro,Beach,70,Summer"]),
            FILES["reviews"]: ("ReviewID,DestinationID,UserID,Rating,ReviewText", [
                "r1,d1,u1,5,Excelente", "r2,d2,u1,4,Bueno", "r3,d3,u2,5,Si", "r4,d4,u3,1,No"]),
            FILES["history"]: ("HistoryID,UserID,DestinationID,VisitDate,ExperienceRating", [
                "h1,u1,d3,2024-01-01,4", "h2,u2,d2,2024-02-01,5", "h3,u2,d4,2024-02-02,4", "h4,u3,d1,2024-02-03,3"]),
        }
        for filename, (header, rows) in content.items():
            (root / filename).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    def test_metrics(self):
        m = rank_metrics(["d1", "d2", "d3"], {"d1", "d3"}, 3)
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertEqual(m["recall"], 1.0)
        self.assertTrue(0 <= m["ndcg"] <= 1)
        with self.assertRaises(ValueError):
            rank_metrics(["d1", "d1"], {"d1"}, 2)

    def test_audit_and_split_and_baseline(self):
        with TemporaryDirectory() as td:
            source = Path(td) / "raw"
            output = Path(td) / "processed"
            source.mkdir()
            self._fixture(source)
            datasets = load_all(source)
            result = audit(datasets)
            self.assertEqual(result["users_unique"], 4)
            self.assertEqual(result["destinations_unique"], 4)
            self.assertEqual(result["pairs_unique_valid"], 8)
            export_processed(datasets, output)
            report = build_split(output, seed=42, cold_fraction=0.25)
            self.assertGreaterEqual(report["cold_users"], 1)
            with (output / "split_manifest.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({(r["user_id"], r["destination_id"]) for r in rows}), 8)
            cold = {r["user_id"] for r in rows if r["split"] == "cold_test"}
            train = {r["user_id"] for r in rows if r["split"] == "train"}
            self.assertFalse(cold & train)
            scores = evaluate_popularity(output, k=2)
            self.assertEqual(scores["k"], 2)
            self.assertTrue((output / "baseline_popularity_metrics.json").is_file())
            # El perfil procesado no guarda columnas de identificación personal.
            self.assertNotIn("Email", (output / "profiles.csv").read_text(encoding="utf-8").splitlines()[0])

    def test_split_reproducible(self):
        with TemporaryDirectory() as td:
            source = Path(td) / "raw"
            output = Path(td) / "processed"
            source.mkdir()
            self._fixture(source)
            export_processed(load_all(source), output)
            build_split(output, seed=3)
            before = (output / "split_manifest.csv").read_bytes()
            build_split(output, seed=3)
            self.assertEqual(before, (output / "split_manifest.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
