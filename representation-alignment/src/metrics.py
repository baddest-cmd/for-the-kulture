"""
Module Name and Role: metrics.py - Component of representation-alignment/src/metrics.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np

def mrr(predictions, labels):
    # Sort predictions descending
    order = np.argsort(-predictions, axis=1)
    ranks = np.empty_like(order)
    for i in range(predictions.shape[0]):
        ranks[i, order[i]] = np.arange(1, predictions.shape[1] + 1)
    
    # We only care about positive labels
    reciprocal_ranks = labels / ranks
    positive_users = np.sum(np.max(labels, axis=1) > 0)
    if positive_users == 0:
        return 0.0
    # Max reciprocal rank per user (assuming at least one positive item)
    mrr_val = np.sum(np.max(reciprocal_ranks, axis=1)) / positive_users
    return float(mrr_val)

def hr_at_k(predictions, labels, k=10):
    order = np.argsort(-predictions, axis=1)
    top_k_indices = order[:, :k]
    
    hits = 0
    for i in range(predictions.shape[0]):
        if np.sum(labels[i, top_k_indices[i]]) > 0:
            hits += 1
    return float(hits / predictions.shape[0]) if predictions.shape[0] > 0 else 0.0

def differentiable_gini(x, epsilon=1e-8, epsilon_denom=1e-6, backend='numpy'):
    """
    Computes a differentiable approximation of the Gini coefficient.
    G(x) = sum_{i=1}^K sum_{j=1}^K sqrt((x_i - x_j)^2 + epsilon) / (2 * K * (sum_{i=1}^K x_i + epsilon_denom))
    """
    if backend == 'jax':
        import jax.numpy as jnp
        math = jnp
    else:
        math = np

    K = x.shape[-1]
    
    # x shape could be (batch, K) or (K,)
    if len(x.shape) == 1:
        diff = x[:, None] - x[None, :]
        abs_diff = math.sqrt(diff**2 + epsilon)
        denom = 2.0 * K * (math.sum(x) + epsilon_denom)
        gini = math.sum(abs_diff) / denom
    else:
        diff = x[:, :, None] - x[:, None, :]
        abs_diff = math.sqrt(diff**2 + epsilon)
        denom = 2.0 * K * (math.sum(x, axis=1) + epsilon_denom)
        gini = math.sum(abs_diff, axis=(1,2)) / denom
        gini = math.mean(gini)
        
    return gini
