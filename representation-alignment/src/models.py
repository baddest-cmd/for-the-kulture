"""
Module Name and Role: models.py - Component of representation-alignment/src/models.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np

# Pure NumPy Implementation (Always Available)
class NumpyTwoTowerModel:
    def __init__(self, hidden_dim, num_prototypes):
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        
        # Simple manual weights dictionary
        self.params = {
            'user_dense_w': np.random.randn(16, hidden_dim) * 0.1,
            'user_dense_b': np.zeros(hidden_dim),
            'item_dense_w': np.random.randn(16, hidden_dim) * 0.1,
            'item_dense_b': np.zeros(hidden_dim),
            'prototypes': np.random.randn(num_prototypes, hidden_dim) * 0.1
        }
        
    def apply(self, params, user_features, item_features):
        u_emb = user_features @ params['user_dense_w'] + params['user_dense_b']
        i_emb = item_features @ params['item_dense_w'] + params['item_dense_b']
        
        u_emb = u_emb / np.linalg.norm(u_emb, axis=1, keepdims=True)
        i_emb = i_emb / np.linalg.norm(i_emb, axis=1, keepdims=True)
        
        predictions = u_emb @ i_emb.T
        return predictions, u_emb, i_emb
        
    def get_prototypes(self, params):
        p = params['prototypes']
        return p / np.linalg.norm(p, axis=1, keepdims=True)

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
            self.prototypes = self.param('prototypes', 
                                         jax.nn.initializers.normal(stddev=0.1), 
                                         (self.num_prototypes, self.hidden_dim))
            
        def __call__(self, user_features, item_features):
            u_emb = self.user_dense(user_features)
            i_emb = self.item_dense(item_features)
            
            # Normalize to project onto unit sphere
            u_emb = u_emb / jnp.linalg.norm(u_emb, axis=1, keepdims=True)
            i_emb = i_emb / jnp.linalg.norm(i_emb, axis=1, keepdims=True)
            
            # Dot product for predicted affinity
            predictions = jnp.dot(u_emb, i_emb.T)
            
            return predictions, u_emb, i_emb

        def get_prototypes(self):
            p = self.prototypes
            return p / jnp.linalg.norm(p, axis=1, keepdims=True)

    TwoTowerModel = FlaxTwoTowerModel
else:
    FlaxTwoTowerModel = None
    TwoTowerModel = NumpyTwoTowerModel
