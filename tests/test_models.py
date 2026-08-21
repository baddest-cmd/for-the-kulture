"""
Module Name and Role: test_models.py - Component of tests/test_models.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kulture.alignment.models import NumpyTwoTowerModel, HAS_JAX


class TestModels(unittest.TestCase):
    def setUp(self):
        self.n_users = 8
        self.n_items = 12
        self.feature_dim = 16
        self.hidden_dim = 8
        self.num_prototypes = 4

        np.random.seed(42)
        self.user_feats = np.random.randn(self.n_users, self.feature_dim).astype(np.float32)
        self.item_feats = np.random.randn(self.n_items, self.feature_dim).astype(np.float32)

    def test_numpy_two_tower_unit_sphere_projection(self):
        """All embeddings and prototypes must have L2 norm = 1.0 on S^(D-1)."""
        model = NumpyTwoTowerModel(
            hidden_dim=self.hidden_dim,
            num_prototypes=self.num_prototypes,
            input_dim=self.feature_dim
        )
        preds, u_emb, i_emb = model.apply(model.params, self.user_feats, self.item_feats)
        prototypes = model.get_prototypes(model.params)

        # 1. Check user embedding norms
        u_norms = np.linalg.norm(u_emb, axis=1)
        np.testing.assert_allclose(u_norms, 1.0, atol=1e-5)

        # 2. Check item embedding norms
        i_norms = np.linalg.norm(i_emb, axis=1)
        np.testing.assert_allclose(i_norms, 1.0, atol=1e-5)

        # 3. Check prototype norms
        p_norms = np.linalg.norm(prototypes, axis=1)
        np.testing.assert_allclose(p_norms, 1.0, atol=1e-5)

        # 4. Predictions shape check
        self.assertEqual(preds.shape, (self.n_users, self.n_items))

    def test_euclidean_cosine_isomorphism_on_sphere(self):
        """
        On S^(D-1), ||x - y||^2 == 2 * (1 - cos(x, y)).
        Minimizing Euclidean distance is isomorphic to maximizing Cosine similarity.
        """
        model = NumpyTwoTowerModel(hidden_dim=8, num_prototypes=4, input_dim=self.feature_dim)
        _, _, i_emb = model.apply(model.params, self.user_feats, self.item_feats)
        prototypes = model.get_prototypes(model.params)

        # Take first item and all prototypes
        item = i_emb[0] # (8,)
        
        # Cosine similarities
        cos_sims = prototypes @ item # (4,)
        # Euclidean squared distances
        euc_sq_dists = np.sum((prototypes - item[None, :]) ** 2, axis=1) # (4,)

        # Check isomorphism formula: ||p - i||^2 = 2 * (1 - cos(p, i))
        expected_dists = 2.0 * (1.0 - cos_sims)
        np.testing.assert_allclose(euc_sq_dists, expected_dists, atol=1e-5)

        # Nearest prototype by Euclidean distance must match highest cosine similarity
        self.assertEqual(int(np.argmin(euc_sq_dists)), int(np.argmax(cos_sims)))

    def test_flax_model_if_available(self):
        """Tests Flax model if JAX is installed in environment."""
        if not HAS_JAX:
            self.skipTest("JAX/Flax not installed in this Python environment.")

        import jax
        import jax.numpy as jnp
        from kulture.alignment.models import FlaxTwoTowerModel

        key = jax.random.PRNGKey(0)
        model = FlaxTwoTowerModel(hidden_dim=self.hidden_dim, num_prototypes=self.num_prototypes)
        params = model.init(key, self.user_feats, self.item_feats)["params"]

        preds, u_emb, i_emb = model.apply({"params": params}, self.user_feats, self.item_feats)
        prototypes = model.get_prototypes(params)

        u_norms = jnp.linalg.norm(u_emb, axis=1)
        i_norms = jnp.linalg.norm(i_emb, axis=1)
        p_norms = jnp.linalg.norm(prototypes, axis=1)

        np.testing.assert_allclose(np.array(u_norms), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.array(i_norms), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.array(p_norms), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
