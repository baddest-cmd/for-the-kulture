"""
Module Name and Role: models.py - Component of src/kulture/alignment/models.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np

# Pure NumPy Implementation (Always Available)
class NumpyTwoTowerModel:
    def __init__(self, hidden_dim: int = 16, num_prototypes: int = 4, input_dim: int = 16):
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.input_dim = input_dim
        
        # Initial weights
        np.random.seed(42)
        self.params = {
            "user_dense_w": np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1,
            "user_dense_b": np.zeros(hidden_dim, dtype=np.float32),
            "item_dense_w": np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1,
            "item_dense_b": np.zeros(hidden_dim, dtype=np.float32),
            "prototypes": np.random.randn(num_prototypes, hidden_dim).astype(np.float32) * 0.1,
        }
        
    def apply(self, params, user_features, item_features, eps: float = 1e-8):
        u_emb = user_features @ params["user_dense_w"] + params["user_dense_b"]
        i_emb = item_features @ params["item_dense_w"] + params["item_dense_b"]
        
        # Spherical L2 projection onto S^(D-1)
        u_emb = u_emb / (np.linalg.norm(u_emb, axis=1, keepdims=True) + eps)
        i_emb = i_emb / (np.linalg.norm(i_emb, axis=1, keepdims=True) + eps)
        
        predictions = u_emb @ i_emb.T
        return predictions, u_emb, i_emb
        
    def get_prototypes(self, params, eps: float = 1e-8):
        p = params["prototypes"]
        return p / (np.linalg.norm(p, axis=1, keepdims=True) + eps)


# Flax Linen Implementation
try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if HAS_JAX:
    class FlaxTwoTowerModel(nn.Module):
        hidden_dim: int
        num_prototypes: int
        
        def setup(self):
            self.user_dense = nn.Dense(self.hidden_dim)
            self.item_dense = nn.Dense(self.hidden_dim)
            
            # Prototypes on the unit sphere
            self.prototypes = self.param(
                "prototypes", 
                jax.nn.initializers.normal(stddev=0.1), 
                (self.num_prototypes, self.hidden_dim)
            )
            
        def __call__(self, user_features, item_features, eps: float = 1e-8):
            u_emb = self.user_dense(user_features)
            i_emb = self.item_dense(item_features)
            
            # Spherical L2 projection onto S^(D-1)
            u_emb = u_emb / (jnp.linalg.norm(u_emb, axis=1, keepdims=True) + eps)
            i_emb = i_emb / (jnp.linalg.norm(i_emb, axis=1, keepdims=True) + eps)
            
            # Cosine affinity via dot product
            predictions = jnp.dot(u_emb, i_emb.T)
            
            return predictions, u_emb, i_emb

        def get_prototypes(self, params=None, eps: float = 1e-8):
            if params is not None and "prototypes" in params:
                p = params["prototypes"]
            else:
                p = self.prototypes
            return p / (jnp.linalg.norm(p, axis=1, keepdims=True) + eps)

    TwoTowerModel = FlaxTwoTowerModel
else:
    FlaxTwoTowerModel = None
    TwoTowerModel = NumpyTwoTowerModel
