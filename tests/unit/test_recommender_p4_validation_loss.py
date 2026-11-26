"""Comprobar BCE real de validacion sin fuga de negativos explícitos."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transport_ai.recommender.data import read_csv, write_csv
from transport_ai.recommender.splits import build_split

class ValidationBceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_csv(self.root / "catalog.csv", ["destination_id", "destination_name", "state", "type", "best_time_to_visit"], [
            {"destination_id": f"d{i}", "destination_name": f"Lugar {i}", "state": "E", "type": ["Beach", "Historical", "Nature", "Adventure", "City"][i % 5], "best_time_to_visit": "Enero"}
            for i in range(10)])
        write_csv(self.root / "profiles.csv", ["user_id", "preferences", "number_of_adults", "number_of_children"], [
            {"user_id": f"u{i}", "preferences": "Beach, Historical", "number_of_adults": "1", "number_of_children": "0"}
            for i in range(15)])
        interactions = []
        for i in range(15):
            for j in range(4):
                interactions.append({"user_id": f"u{i}", "destination_id": f"d{(i+j) % 10}", "source": "reviews", "rating": "5", "visit_date": ""})
            for j in range(4, 7):
                interactions.append({"user_id": f"u{i}", "destination_id": f"d{(i+j) % 10}", "source": "reviews", "rating": "2", "visit_date": ""})
        write_csv(self.root / "interactions.csv", ["user_id", "destination_id", "source", "rating", "visit_date"], interactions)
        self.report = build_split(self.root, seed=42, cold_fraction=0.1)

    def test_heldout_nonpositives_do_not_enter_training(self):
        manifest, _ = read_csv(self.root / "split_manifest.csv")
        val_negative = {(r["user_id"], r["destination_id"]) for r in manifest if r["split"] == "validation_nonpositive"}
        self.assertTrue(val_negative)
        self.assertEqual(len(val_negative), self.report["validation_bce_nonpositive_count"])
        self.assertEqual(len(val_negative), self.report["validation_bce_positive_count"])
        from transport_ai.recommender.neural import build_bundle
        bundle = build_bundle(self.root)
        train_pairs = {(u,d) for u,d,_ in bundle["examples"]}
        self.assertFalse(train_pairs & val_negative)
        self.assertFalse({r["user_id"] for r in manifest if r["split"] == "cold_test"} &
                         {u for u,d in val_negative})

    def test_curves_contain_actual_validation_bce(self):
        from transport_ai.recommender.neural import build_bundle
        from transport_ai.recommender.evaluation import load_context
        from transport_ai.recommender.train import fit_one, DEFAULT_CONFIG
        model, report = fit_one(build_bundle(self.root), load_context(self.root), DEFAULT_CONFIG,
                                seed=42, epochs=2, k=1)
        self.assertTrue(report["curves"])
        self.assertEqual(report["validation_bce_sample_size"],
                         2 * self.report["validation_bce_nonpositive_count"])
        self.assertTrue(all("train_bce" in p and "validation_bce" in p and "validation_ndcg_at_k" in p
                            for p in report["curves"]))
        self.assertTrue(all(p["validation_bce"] >= 0 for p in report["curves"]))

if __name__ == "__main__":
    unittest.main()
