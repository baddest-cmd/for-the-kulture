"""
Module Name and Role: test_metrics.py - Component of tests/test_metrics.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kulture.alignment.metrics import differentiable_gini, mrr, hr_at_k


class TestMetrics(unittest.TestCase):
    def test_gini_uniform_distribution(self):
        """Uniform exposure should yield approximately 0.0 Gini coefficient."""
        x = np.ones(10, dtype=np.float32) * 5.0
        g = differentiable_gini(x, epsilon=1e-8)
        self.assertAlmostEqual(float(g), 0.0, places=3)

    def test_gini_extreme_monopoly(self):
        """One item receives all exposure, rest receive zero."""
        K = 10
        x = np.zeros(K, dtype=np.float32)
        x[0] = 100.0
        g = differentiable_gini(x, epsilon=1e-8)
        # Theoretical discrete Gini for one non-zero out of K items is (K - 1) / K = 0.9
        expected = (K - 1) / K
        self.assertAlmostEqual(float(g), expected, delta=0.05)

    def test_gini_zero_sum_safety(self):
        """Zero exposure vector should not raise ZeroDivisionError."""
        x = np.zeros(10, dtype=np.float32)
        g = differentiable_gini(x, epsilon=1e-8, epsilon_denom=1e-6)
        self.assertTrue(np.isfinite(g))

    def test_gini_batch_2d(self):
        """Batch of exposure vectors (batch_size, K)."""
        x_batch = np.array([
            [1.0, 1.0, 1.0, 1.0],      # Uniform
            [10.0, 0.0, 0.0, 0.0],     # Skewed
        ], dtype=np.float32)
        g = differentiable_gini(x_batch)
        self.assertTrue(0.0 <= g <= 1.0)

    def test_mrr_perfect_ranking(self):
        """When positive item is ranked #1 for all users, MRR should be 1.0."""
        predictions = np.array([
            [10.0, 5.0, 1.0],
            [9.0, 2.0, 0.0]
        ])
        labels = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        self.assertEqual(mrr(predictions, labels), 1.0)

    def test_mrr_second_rank(self):
        """When positive item is ranked #2, MRR should be 0.5."""
        predictions = np.array([
            [5.0, 10.0, 1.0]
        ])
        labels = np.array([
            [1.0, 0.0, 0.0]
        ])
        self.assertEqual(mrr(predictions, labels), 0.5)

    def test_hr_at_k(self):
        """Check hit rate at top-1 and top-2."""
        predictions = np.array([
            [10.0, 5.0, 1.0],  # Item 0 is rank 1
            [5.0, 10.0, 1.0]   # Item 1 is rank 1, Item 0 is rank 2
        ])
        labels = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        self.assertEqual(hr_at_k(predictions, labels, k=1), 0.5)
        self.assertEqual(hr_at_k(predictions, labels, k=2), 1.0)


if __name__ == "__main__":
    unittest.main()
