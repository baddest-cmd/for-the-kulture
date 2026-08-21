import unittest
import numpy as np

from kulture.postprocessing.scruf_d import (
    MyopicExploiterAgent,
    CausalArbiterAgent,
    AdversarialPreserverAgent,
    ScrufdRecommender,
)


class TestScrufd(unittest.TestCase):
    def setUp(self):
        self.item_genres = [
            "Pop", "Pop", "Pop", "Pop", "Pop",           # 0..4 Non-cultural
            "Gqom", "Lekompo", "Maskandi", "SA House",   # 5..8 Cultural
            "Pop"                                         # 9 Non-cultural
        ]
        # Baseline model strongly favors generic pop (items 0..4)
        self.predictions = np.array([
            0.95, 0.90, 0.88, 0.85, 0.80,
            0.50, 0.45, 0.40, 0.35,
            0.75
        ], dtype=np.float32)
        self.popularities = np.array([
            1.0, 0.9, 0.8, 0.7, 0.6,
            0.3, 0.2, 0.2, 0.3,
            0.5
        ], dtype=np.float32)

    def test_myopic_exploiter_picks_top_predictions(self):
        """Myopic agent should strictly follow raw predictions."""
        agent = MyopicExploiterAgent()
        scores = agent.score(self.predictions)
        top_3 = np.argsort(-scores)[:3]
        self.assertEqual(list(top_3), [0, 1, 2])

    def test_causal_arbiter_penalizes_popularity_under_high_tax(self):
        """High tau_c should penalize high popularity items."""
        arbiter = CausalArbiterAgent(beta=0.5)
        # With tau_c = 1.0, penalty = 0.5 * 1.0 * popularity
        scores_high_tax = arbiter.score(self.predictions, context_tax=1.0, item_popularities=self.popularities)
        scores_zero_tax = arbiter.score(self.predictions, context_tax=0.0, item_popularities=self.popularities)

        # High tax should penalize item 0 (pop=1.0, penalty=0.5 -> score=0.45)
        self.assertLess(scores_high_tax[0], scores_zero_tax[0])

    def test_adversarial_preserver_scores_cultural_items(self):
        """Preserver should score cultural subgenres positively."""
        preserver = AdversarialPreserverAgent(
            target_subgenres=["Gqom", "Lekompo", "Maskandi", "SA House"],
            quota_weight=1.0
        )
        scores = preserver.score(self.item_genres)
        self.assertEqual(scores[0], 0.0) # Pop
        self.assertEqual(scores[5], 1.0) # Gqom
        self.assertEqual(scores[6], 1.0) # Lekompo

    def test_scrufd_recommender_enforces_quota(self):
        """Recommender must enforce minimum cultural quota in recommendation slate."""
        recommender = ScrufdRecommender(
            w_exploit=0.4,
            w_arbiter=0.3,
            w_preserve=0.3,
            min_cultural_quota=2,
            k=4
        )

        result = recommender.recommend(
            user_id=1,
            predictions=self.predictions,
            context_tax=0.8,
            item_genres=self.item_genres,
            item_popularities=self.popularities
        )

        self.assertEqual(len(result["slate_indices"]), 4)
        self.assertTrue(result["cultural_quota_met"])
        self.assertGreaterEqual(result["cultural_item_count"], 2)

    def test_scrufd_recommender_zero_quota(self):
        """Recommender should function normally when min_cultural_quota is 0."""
        recommender = ScrufdRecommender(
            w_exploit=0.8,
            w_arbiter=0.2,
            w_preserve=0.0,
            min_cultural_quota=0,
            k=3
        )

        result = recommender.recommend(
            user_id=2,
            predictions=self.predictions,
            context_tax=0.0,
            item_genres=self.item_genres,
            item_popularities=self.popularities
        )

        self.assertEqual(len(result["slate_indices"]), 3)
        self.assertTrue(result["cultural_quota_met"])
        # Should match top myopic/arbiter choices: 0, 1, 2
        self.assertEqual(list(result["slate_indices"]), [0, 1, 2])

    def test_scrufd_recommender_insufficient_cultural_items(self):
        """Recommender handles case when available cultural items are less than required quota."""
        # Only 1 cultural item in list
        limited_genres = ["Pop"] * 9 + ["Gqom"]
        recommender = ScrufdRecommender(
            w_exploit=0.5,
            w_arbiter=0.3,
            w_preserve=0.2,
            min_cultural_quota=3,  # Quota is 3, but only 1 exists
            k=4
        )

        result = recommender.recommend(
            user_id=3,
            predictions=self.predictions,
            context_tax=0.5,
            item_genres=limited_genres,
            item_popularities=self.popularities
        )

        self.assertEqual(len(result["slate_indices"]), 4)
        self.assertFalse(result["cultural_quota_met"])
        self.assertEqual(result["cultural_item_count"], 1)


if __name__ == "__main__":
    unittest.main()
