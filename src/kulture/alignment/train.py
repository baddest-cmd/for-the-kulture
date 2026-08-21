"""
Module Name and Role: train.py - Component of src/kulture/alignment/train.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from .metrics import differentiable_gini, mrr, hr_at_k
from .models import TwoTowerModel, FlaxTwoTowerModel, NumpyTwoTowerModel, HAS_JAX
from .utils import MusicDatasetLoader

if HAS_JAX:
    import jax
    import jax.numpy as jnp
    import optax


def compute_loss(
    params,
    model,
    user_feats,
    item_feats,
    interactions,
    lambda_proto: float = 0.1,
    lambda_gini: float = 0.1,
    backend: str = "numpy"
):
    """
    Computes total joint loss across reconstruction, prototype clustering, and exposure Gini.
    """
    if backend == "jax":
        math = jnp
    else:
        math = np

    # 1. Forward Pass
    if backend == "jax":
        predictions, u_emb, i_emb = model.apply({"params": params}, user_feats, item_feats)
        prototypes = params["prototypes"]
        prototypes = prototypes / (math.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
    else:
        predictions, u_emb, i_emb = model.apply(params, user_feats, item_feats)
        prototypes = model.get_prototypes(params)

    # 2. Reconstruction MSE Loss
    loss_recon = math.mean((predictions - interactions) ** 2)

    # 3. Prototype Clustering & Separation Loss
    # Cosine similarity matrix between items and prototypes
    sim = math.dot(i_emb, prototypes.T) # shape: (n_items, n_prototypes)
    max_sim = math.max(sim, axis=1)     # shape: (n_items,)
    loss_proto_cluster = math.mean(1.0 - max_sim)

    # Prototype Separation (push centroids apart)
    proto_sim = math.dot(prototypes, prototypes.T)
    mask = 1.0 - math.eye(prototypes.shape[0])
    loss_proto_sep = math.mean(proto_sim * mask)
    loss_proto = loss_proto_cluster + 0.5 * loss_proto_sep

    # 4. Smooth Differentiable Gini Exposure Loss
    item_exposure = math.sum(predictions, axis=0) # shape: (n_items,)
    item_exposure = math.maximum(item_exposure, 0.0) + 1e-5
    loss_gini = differentiable_gini(item_exposure, backend=backend)

    # Total joint loss
    total_loss = loss_recon + lambda_proto * loss_proto + lambda_gini * loss_gini
    return total_loss, (loss_recon, loss_proto, loss_gini, predictions)


def finite_difference_grad(params, model, user_feats, item_feats, interactions, lp=0.1, lg=0.1, epsilon=1e-4):
    """Central finite difference gradients for pure NumPy fallback."""
    grads = {}
    for k, v in params.items():
        flat_v = v.flatten()
        flat_grad = np.zeros_like(flat_v)
        
        for i in range(len(flat_v)):
            original = flat_v[i]
            
            flat_v[i] = original + epsilon
            params_plus = params.copy()
            params_plus[k] = flat_v.reshape(v.shape)
            loss_plus, _ = compute_loss(params_plus, model, user_feats, item_feats, interactions, lp, lg, backend="numpy")
            
            flat_v[i] = original - epsilon
            params_minus = params.copy()
            params_minus[k] = flat_v.reshape(v.shape)
            loss_minus, _ = compute_loss(params_minus, model, user_feats, item_feats, interactions, lp, lg, backend="numpy")
            
            flat_grad[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
            flat_v[i] = original
            
        grads[k] = flat_grad.reshape(v.shape)
    return grads


def train(
    lambda_proto: float = 0.1,
    lambda_gini: float = 0.1,
    epochs: int = 10,
    learning_rate: float = 0.05,
    seed: int = 42,
    quiet: bool = False
):
    """
    Trains the Spherical Two-Tower Model with joint alignment loss.
    """
    loader = MusicDatasetLoader(seed=seed)
    train_interactions, test_interactions, user_features, item_features = loader.load_or_generate_data()

    hidden_dim = 16
    num_prototypes = 4

    if HAS_JAX:
        if not quiet:
            print("Using JAX/Flax backend with stateless PRNG splitting.")
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        
        model = FlaxTwoTowerModel(hidden_dim=hidden_dim, num_prototypes=num_prototypes)
        params = model.init(subkey, user_features, item_features)["params"]
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        @jax.jit
        def train_step(params, opt_state, u_f, i_f, interactions):
            def loss_fn(p):
                return compute_loss(
                    p, model, u_f, i_f, interactions,
                    lambda_proto=lambda_proto, lambda_gini=lambda_gini, backend="jax"
                )
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, metrics

        for epoch in range(epochs):
            params, opt_state, loss, aux_metrics = train_step(
                params, opt_state, user_features, item_features, train_interactions
            )
            loss_recon, loss_proto, loss_gini, predictions = aux_metrics
            mrr_val = mrr(np.array(predictions), test_interactions)

            if not quiet:
                print(
                    f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Recon: {loss_recon:.4f} | "
                    f"Proto: {loss_proto:.4f} | Gini: {loss_gini:.4f} | MRR: {mrr_val:.4f}"
                )
    else:
        if not quiet:
            print("Using pure-NumPy fallback with finite-difference gradients.")
        model = NumpyTwoTowerModel(hidden_dim=hidden_dim, num_prototypes=num_prototypes)
        params = model.params

        for epoch in range(epochs):
            loss, aux_metrics = compute_loss(
                params, model, user_features, item_features, train_interactions,
                lambda_proto=lambda_proto, lambda_gini=lambda_gini, backend="numpy"
            )
            loss_recon, loss_proto, loss_gini, predictions = aux_metrics
            grads = finite_difference_grad(
                params, model, user_features, item_features, train_interactions,
                lp=lambda_proto, lg=lambda_gini
            )

            for k in params.keys():
                params[k] -= learning_rate * grads[k]

            mrr_val = mrr(predictions, test_interactions)
            if not quiet:
                print(
                    f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Recon: {loss_recon:.4f} | "
                    f"Proto: {loss_proto:.4f} | Gini: {loss_gini:.4f} | MRR: {mrr_val:.4f}"
                )

    return params, model, HAS_JAX
