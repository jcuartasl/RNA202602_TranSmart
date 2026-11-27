import unittest
from transport_ai.recommender.reranking import rerank_by_preferences

class HybridTest(unittest.TestCase):
    def test_preference_bonus(self):
        catalog = {
            "a": {"type": "Beach"},
            "b": {"type": "City"}
        }
        scores = {"a": 0.40, "b": 0.41}
        result = rerank_by_preferences(scores, catalog, "Beaches", 0.02)
        self.assertAlmostEqual(result["a"], 0.42)
        self.assertAlmostEqual(result["b"], 0.41)
        self.assertEqual(scores["a"], 0.40)

    def test_zero_bonus_preserves_scores(self):
        catalog = {"a": {"type": "Beach"}}
        scores = {"a": 0.40}
        self.assertEqual(
            rerank_by_preferences(scores, catalog, "Beaches", 0),
            scores
        )
