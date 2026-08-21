"""
Module Name and Role: test_analysis.py - Component of tests/test_analysis.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kulture.analysis.pipeline import CulturalAnalysisPipeline


class TestCulturalAnalysisPipeline(unittest.TestCase):
    def setUp(self):
        # Create minimal mock dataframe with required schema
        self.mock_df = pd.DataFrame({
            "discovery_method": [
                "My streaming app's recommendations/algorithm",
                "My streaming app's recommendations/algorithm",
                "Radio",
                "Social media (TikTok, Instagram, etc.)"
            ] * 10,
            "local_discovery_method": [
                "Social media (TikTok, Instagram, etc.)",
                "Social media (TikTok, Instagram, etc.)",
                "Radio",
                "Social media (TikTok, Instagram, etc.)"
            ] * 10,
            "rec_system_rating": [6.0, 5.0, 2.0, 4.0] * 10,
            "age_band": ["18-24", "25-34", "35-44", "18-24"] * 10,
            "used_ai_features": ["Yes", "No", "No", "Yes"] * 10,
            "challenge_following_local_artists": [
                "Hard to discover new songs on platforms",
                "Lack of consistency and promotion",
                "No radio play for underground acts",
                "Hard to find tour dates on social media"
            ] * 10
        })

    def test_inversion_shift(self):
        pipeline = CulturalAnalysisPipeline(data_path=self.mock_df)
        shift = pipeline.run_inversion_shift_analysis()
        self.assertIn("algo_reliance_shift_pp", shift)
        self.assertIn("social_discovery_shift_pp", shift)
        # Algorithm should have dropped, social media should have increased
        self.assertLess(shift["algo_reliance_shift_pp"], 0.0)
        self.assertGreater(shift["social_discovery_shift_pp"], 0.0)

    def test_non_parametric_tests(self):
        pipeline = CulturalAnalysisPipeline(data_path=self.mock_df)
        res = pipeline.run_non_parametric_tests()
        self.assertIn("age_band_kruskal", res)
        self.assertIn("ai_adoption_mannwhitney", res)

    def test_lda_topic_modeling(self):
        pipeline = CulturalAnalysisPipeline(data_path=self.mock_df)
        topics_res = pipeline.run_lda_topic_modeling(n_topics=2)
        self.assertEqual(len(topics_res["topics"]), 2)


if __name__ == "__main__":
    unittest.main()
