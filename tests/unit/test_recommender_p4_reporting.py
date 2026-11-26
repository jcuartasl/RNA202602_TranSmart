"""Pruebas autocontenidas para auditoría, reporte, figuras y campaña M3."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transport_ai.recommender.audits import catalog_audit, ranking_duplicate_names
from transport_ai.recommender.business import campaign_scenario
from transport_ai.recommender.data import FILES, export_processed, load_all
from transport_ai.recommender.splits import build_split


class ReportingTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "raw"
        source.mkdir()
        types = ["Beach", "Historical", "Nature", "Adventure", "City"]
        dest = [f"d{i},Destino {i},Estado {i},{types[(i-1) % 5]},0,Enero" for i in range(1, 11)]
        dest[1] = "d2,Destino 1,Estado 1,Historical,0,Enero"  # posible colisión de nombre, pero no de tipo
        users = [f"u{i},Usuario{i},mail{i}@ejemplo.com,{('Beaches; Historical' if i % 2 else 'Nature; Adventure')},X,2,0"
                 for i in range(1, 13)]
        rev = [f"r{i}_{j},u{i},d{((i+j) % 10) + 1},5,ok" for i in range(1, 13) for j in range(3)]
        rev += [f"rn{i},u{i},d{((i+6) % 10) + 1},1,mal" for i in range(1, 13)]
        hist = ["h1,u1,d10,2024-01-01,2"]
        contents = {
            FILES["users"]: ("UserID,Name,Email,Preferences,Gender,NumberOfAdults,NumberOfChildren", users),
            FILES["destinations"]: ("DestinationID,Name,State,Type,Popularity,BestTimeToVisit", dest),
            FILES["reviews"]: ("ReviewID,UserID,DestinationID,Rating,ReviewText", rev),
            FILES["history"]: ("HistoryID,UserID,DestinationID,VisitDate,ExperienceRating", hist),
        }
        for name, (header, rows) in contents.items():
            (source / name).write_text("\n".join([header, *rows]) + "\n", encoding="utf8")
        self.processed = self.root / "processed"
        self.results = self.root / "experiments" / "m3" / "diagnostics"
        self.neural_results = self.root / "experiments" / "m3" / "neural"
        self.weights = self.root / "weights"
        self.knn = self.root / "knn.json"
        export_processed(load_all(source), self.processed)
        build_split(self.processed, seed=42, cold_fraction=0.1)

    def test_audit_and_campaign(self):
        a = catalog_audit(self.processed, self.results / "catalog.json")
        self.assertGreaterEqual(a["same_name"]["repeated_groups"], 1)
        self.assertEqual(a["same_name_and_state"]["repeated_groups"], 1)
        names = ranking_duplicate_names([
            {"destination_id": "x", "destination_name": "Taj Mahal"},
            {"destination_id": "y", "destination_name": " Taj  Mahal "},
        ])
        self.assertEqual(names["distinct_names"], 1)
        c = campaign_scenario(self.processed, self.results / "campaign.json", destination_type="beach",
                              exposure_rate=0.5, assumed_conversion_rate=0.1)
        self.assertEqual(c["eligible_profiles"], 6)
        self.assertAlmostEqual(c["expected_conversions"], 0.3)
        with self.assertRaises(ValueError):
            campaign_scenario(self.processed, self.results / "bad.json", exposure_rate=2)

    def test_report_and_plots_without_using_test(self):
        try:
            import torch  # noqa
            import matplotlib  # noqa
        except ImportError:
            self.skipTest("matplotlib/torch no disponibles")
        from transport_ai.recommender.classical import train_item_knn
        from transport_ai.recommender.train import train_seeds
        from transport_ai.recommender.reporting import generate_report
        from transport_ai.recommender.plots import make_figures

        train_item_knn(self.processed, self.knn, neighbors=5)
        train_seeds(self.processed, self.neural_results, self.weights, seeds=1, epochs=2, k=5)
        rep = generate_report(self.processed, self.knn, self.weights / "best.pt",
                              self.results, split="validation", ks=(1, 3, 5))
        self.assertEqual(rep["rows_written"], 12)
        self.assertEqual(set(rep["metrics_at_5"]), {"popularity", "item-knn", "content", "neural"})
        self.assertTrue(Path(rep["report_json"]).is_file())
        figures = make_figures(self.processed, self.results, self.weights / "best.pt",
                               self.root / "figures", split="validation", training_results=self.neural_results)
        self.assertGreaterEqual(figures["figure_count"], 5)
        self.assertEqual(figures["training_results_source"], str(self.neural_results))
        self.assertTrue(all(Path(p).is_file() for p in figures["figures"]))

    def test_entity_friendly_k_and_neural_curves_use_same_experiment(self):
        try:
            import torch  # noqa
            import matplotlib  # noqa
        except ImportError:
            self.skipTest("matplotlib/torch no disponibles")
        from transport_ai.recommender.classical import train_item_knn
        from transport_ai.recommender.train import train_seeds
        from transport_ai.recommender.reporting import generate_report
        from transport_ai.recommender.plots import make_figures
        train_item_knn(self.processed, self.knn, neighbors=3)
        separate = self.root / "results" / "other_experiment"
        train_seeds(self.processed, separate, self.weights, seeds=1, epochs=2, k=1)
        rep = generate_report(self.processed, self.knn, self.weights / "best.pt",
                              self.results, split="validation", ks=(1, 2, 3))
        self.assertEqual(set(rep["metrics_by_k"]), {"1", "2", "3"})
        figures = make_figures(self.processed, self.results, self.weights / "best.pt",
                               self.root / "separate_figures", split="validation", training_results=separate)
        self.assertEqual(figures["training_results_source"], str(separate))
        self.assertEqual(figures["figure_count"], 9)
        self.assertTrue(any(str(p).endswith("m3_10_curvas_bce_train_validation.png")
                            for p in figures["figures"]))
        self.assertTrue((self.root / "separate_figures" / "m3_07_boxplot_semillas.png").is_file())


if __name__ == "__main__":
    unittest.main()
