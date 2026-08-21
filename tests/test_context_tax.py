"""
Module Name and Role: test_context_tax.py - Component of tests/test_context_tax.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kulture.postprocessing.context_tax import ContextTaxCalculator


class TestContextTax(unittest.TestCase):
    def test_individual_tax_formula(self):
        """tau_c = S_global - D_local."""
        # Case 1: High global satisfaction, zero local discovery
        tax1 = ContextTaxCalculator.compute_individual_tax(1.0, 0.0)
        self.assertEqual(tax1, 1.0)

        # Case 2: Aligned satisfaction and discovery
        tax2 = ContextTaxCalculator.compute_individual_tax(0.8, 0.8)
        self.assertAlmostEqual(tax2, 0.0)

        # Case 3: Low satisfaction, high discovery
        tax3 = ContextTaxCalculator.compute_individual_tax(0.2, 0.9)
        self.assertAlmostEqual(tax3, -0.7)

    def test_tax_bounds(self):
        """tau_c must always fall within [-1.0, 1.0]."""
        sats = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        discs = np.array([1.5, 1.0, 0.5, 0.0, -0.5])
        taxes = ContextTaxCalculator.compute_individual_tax(sats, discs)

        self.assertTrue(np.all(taxes >= -1.0))
        self.assertTrue(np.all(taxes <= 1.0))

    def test_empirical_scoring_with_mock_df(self):
        """Check empirical scoring on mock survey dataframe."""
        mock_df = pd.DataFrame({
            "rec_system_rating": [7, 4, 1, np.nan],
            "discovery_method": [
                "My streaming app's recommendations/algorithm",
                "My streaming app's recommendations/algorithm",
                "Radio",
                "Social media (TikTok, Instagram, etc.)"
            ],
            "local_discovery_method": [
                "Social media (TikTok, Instagram, etc.)",       # Hopper
                "My streaming app's recommendations/algorithm", # Native
                "Radio",
                "Social media (TikTok, Instagram, etc.)"
            ]
        })

        calc = ContextTaxCalculator(data_path=mock_df)
        scored = calc.compute_empirical_scores()

        self.assertIn("context_tax", scored.columns)
        self.assertIn("is_platform_hopper", scored.columns)

        # First row: S=1.0, D=0.0 -> tau_c = +1.0, Hopper = 1
        self.assertAlmostEqual(scored.loc[0, "context_tax"], 1.0)
        self.assertEqual(scored.loc[0, "is_platform_hopper"], 1)

        # Second row: S=0.5, D=1.0 -> tau_c = -0.5, Hopper = 0
        self.assertAlmostEqual(scored.loc[1, "context_tax"], -0.5)
        self.assertEqual(scored.loc[1, "is_platform_hopper"], 0)


if __name__ == "__main__":
    unittest.main()
