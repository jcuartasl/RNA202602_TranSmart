"""Pruebas de normalización sin acceso a Kaggle ni modelos entrenados."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transport_ai.recommender.data import write_csv, read_csv
from transport_ai.recommender.entities import prepare_entities, report_semantic_overlap
from transport_ai.recommender.splits import build_split


class EntityTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / "by_id"
        self.clean = self.root / "by_entity"
        write_csv(self.raw / "catalog.csv", ["destination_id", "destination_name", "state", "type", "best_time_to_visit"], [
            {"destination_id": "1", "destination_name": "Taj Mahal", "state": "Uttar Pradesh", "type": "Historical", "best_time_to_visit": "Winter"},
            {"destination_id": "2", "destination_name": " Taj   Mahal ", "state": "Uttar Pradesh", "type": "Historical", "best_time_to_visit": "Winter"},
            {"destination_id": "3", "destination_name": "Goa Beaches", "state": "Goa", "type": "Beach", "best_time_to_visit": "Summer"},
            {"destination_id": "4", "destination_name": "Kerala Forest", "state": "Kerala", "type": "Nature", "best_time_to_visit": "Summer"},
        ])
        write_csv(self.raw / "profiles.csv", ["user_id", "preferences", "number_of_adults", "number_of_children"], [
            {"user_id": "u1", "preferences": "Nature", "number_of_adults": "1", "number_of_children": "0"},
            {"user_id": "u2", "preferences": "Historical", "number_of_adults": "2", "number_of_children": "0"},
        ])
        write_csv(self.raw / "interactions.csv", ["user_id", "destination_id", "source", "rating", "visit_date"], [
            {"user_id": "u1", "destination_id": "1", "source": "reviews", "rating": "5", "visit_date": ""},
            {"user_id": "u1", "destination_id": "2", "source": "reviews", "rating": "1", "visit_date": ""},
            {"user_id": "u1", "destination_id": "3", "source": "reviews", "rating": "5", "visit_date": ""},
            {"user_id": "u1", "destination_id": "4", "source": "history", "rating": "5", "visit_date": "2024-01-01"},
            {"user_id": "u2", "destination_id": "2", "source": "reviews", "rating": "4", "visit_date": ""},
        ])
        # Forzamos deliberadamente fuga lógica con IDs distintos del mismo lugar.
        write_csv(self.raw / "split_manifest.csv", ["user_id", "destination_id", "split", "positive"], [
            {"user_id": "u1", "destination_id": "1", "split": "train", "positive": "true"},
            {"user_id": "u1", "destination_id": "2", "split": "validation", "positive": "true"},
            {"user_id": "u1", "destination_id": "3", "split": "test", "positive": "true"},
        ])

    def test_semantic_overlap_and_safe_separate_processing(self):
        before = report_semantic_overlap(self.raw)
        self.assertEqual(before["logical_catalog_size"], 3)
        self.assertEqual(before["overlap_by_split"]["validation"]["heldout_logical_places_already_seen"], 1)
        result = prepare_entities(self.raw, self.clean)
        self.assertEqual(result["logical_entities_by_name_state_type"], 3)
        self.assertEqual(result["entity_user_pairs_conflicting_positive_nonpositive"], 1)
        self.assertEqual(result["destination_user_pairs_after_grouping"], 4)
        old_catalog, _ = read_csv(self.raw / "catalog.csv")
        new_catalog, _ = read_csv(self.clean / "catalog.csv")
        self.assertEqual(len(old_catalog), 4)
        self.assertEqual(len(new_catalog), 3)
        self.assertTrue((self.clean / "entity_id_map.csv").exists())
        report = build_split(self.clean, seed=42, cold_fraction=0.0)
        self.assertEqual(report["conflicted_pairs"], 1)
        self.assertEqual(report["counts_by_split"]["excluded_conflict"], 1)

    def test_no_in_place_modification(self):
        with self.assertRaises(ValueError):
            prepare_entities(self.raw, self.raw)


if __name__ == "__main__":
    unittest.main()
