"""
Module Name and Role: metrics.py - Component of src/kulture/alignment/metrics.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from typing import Union
import numpy as np


def mrr(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    order = np.argsort(-predictions, axis=1)
    ranks = np.empty_like(order)
    for i in range(predictions.shape[0]):
        ranks[i, order[i]] = np.arange(1, predictions.shape[1] + 1)
    
    reciprocal_ranks = labels / ranks
    positive_users = np.sum(np.max(labels, axis=1) > 0)
    if positive_users == 0:
        return 0.0
    return float(np.sum(np.max(reciprocal_ranks, axis=1)) / positive_users)


def hr_at_k(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    Computes Hit Rate at K (HR@K).
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    order = np.argsort(-predictions, axis=1)
    top_k_indices = order[:, :k]
    
    hits = 0
    for i in range(predictions.shape[0]):
        if np.sum(labels[i, top_k_indices[i]]) > 0:
            hits += 1
    return float(hits / predictions.shape[0]) if predictions.shape[0] > 0 else 0.0


def differentiable_gini(
    x: Union[np.ndarray, "jax.numpy.ndarray"],
    epsilon: float = 1e-8,
    epsilon_denom: float = 1e-6,
    backend: str = "numpy"
) -> Union[float, "jax.numpy.ndarray"]:
    """
    Computes a smooth differentiable approximation of the Gini coefficient.
    
    Formula:
        G(x) = sum_{i=1}^K sum_{j=1}^K sqrt((x_i - x_j)^2 + epsilon) / (2 * K * (sum_{i=1}^K x_i + epsilon_denom))
        
    Guarantees C^inf smoothness at x_i = x_j and guards against zero-sum exposures.
    """
    if backend == "jax":
        import jax.numpy as jnp
        math = jnp
    else:
        math = np

    K = x.shape[-1]
    
    # 1D exposure vector (K,)
    if len(x.shape) == 1:
        diff = x[:, None] - x[None, :]
        abs_diff = math.sqrt(diff**2 + epsilon)
        denom = 2.0 * K * (math.sum(x) + epsilon_denom)
        gini = math.sum(abs_diff) / denom
    else:
        # 2D batch of exposures (batch, K)
        diff = x[:, :, None] - x[:, None, :]
        abs_diff = math.sqrt(diff**2 + epsilon)
        denom = 2.0 * K * (math.sum(x, axis=1) + epsilon_denom)
        gini_per_batch = math.sum(abs_diff, axis=(1, 2)) / denom
        gini = math.mean(gini_per_batch)
        
    return gini
