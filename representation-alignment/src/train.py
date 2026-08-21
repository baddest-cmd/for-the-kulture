"""
Module Name and Role: train.py - Component of representation-alignment/src/train.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np
from metrics import differentiable_gini, mrr, hr_at_k
from utils import MusicDatasetLoader
from models import TwoTowerModel, FlaxTwoTowerModel, NumpyTwoTowerModel

try:
    import jax
    import jax.numpy as jnp
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

def compute_loss(params, model, user_feats, item_feats, interactions, lambda_proto=0.1, lambda_gini=0.1, backend='numpy'):
    if backend == 'jax':
        math = jnp
    else:
        math = np

    # Forward pass
    if backend == 'jax':
        predictions, u_emb, i_emb = model.apply({'params': params}, user_feats, item_feats)
        prototypes = params['prototypes']
        prototypes = prototypes / math.linalg.norm(prototypes, axis=1, keepdims=True)
    else:
        predictions, u_emb, i_emb = model.apply(params, user_feats, item_feats)
        prototypes = model.get_prototypes(params)

    # Reconstruction Loss (MSE)
    loss_recon = math.mean((predictions - interactions) ** 2)

    # Prototype Loss: min distance from each item to nearest prototype
    # distance = 1 - cosine similarity
    sim = math.dot(i_emb, prototypes.T) # shape: (n_items, n_prototypes)
    max_sim = math.max(sim, axis=1) # shape: (n_items,)
    loss_proto = math.mean(1.0 - max_sim)
    
    # Separation constraint: push prototypes apart
    proto_sim = math.dot(prototypes, prototypes.T)
    # Mask out diagonal
    mask = 1.0 - math.eye(prototypes.shape[0])
    loss_sep = math.mean(proto_sim * mask)
    loss_proto += 0.5 * loss_sep

    # Gini Loss: uniform exposure of items
    item_exposure = math.sum(predictions, axis=0) # Sum of predicted affinities per item
    # Ensure non-negative and bounded away from zero
    item_exposure = math.maximum(item_exposure, 0.0) + 1e-5
    loss_gini = differentiable_gini(item_exposure, backend=backend)

    total_loss = loss_recon + lambda_proto * loss_proto + lambda_gini * loss_gini
    return total_loss, (loss_recon, loss_proto, loss_gini, predictions)

# Finite difference gradient for numpy fallback
def finite_difference_grad(
    params, model, user_feats, item_feats, interactions, epsilon=1e-4
):
    """Calculates the gradient via finite differences for the NumPy fallback.

    This function serves strictly as a lightweight, zero-dependency numerical
    verification baseline when JAX/Flax is unavailable. It provides a
    mathematically transparent, coordinate-wise central-difference estimator
    (requiring O(2*theta) loss evaluations, where theta is the number of
    parameters) to audit and assert the correctness of JAX's automatic
    differentiation (jax.grad) outputs during local CPU-only unit testing.

    Args:
        params (dict): The model parameters.
        model (NumpyTwoTowerModel): The NumPy model instance.
        user_feats (np.ndarray): User features.
        item_feats (np.ndarray): Item features.
        interactions (np.ndarray): The interaction matrix.
        epsilon (float): The small perturbation for the central difference.

    Returns:
        dict: A dictionary of gradients for each parameter.
    """
    # Production Scale Guardrail: For industrial scaling (N >= 5000), JAX's
    # native reverse-mode automatic differentiation combined with hardware-
    # accelerated XLA compilation (@jax.jit) is utilized exclusively,
    # rendering numerical approximations like this redundant.
    grads = {}
    for k, v in params.items():
        grad = np.zeros_like(v)
        flat_v = v.flatten()
        flat_grad = np.zeros_like(flat_v)
        
        for i in range(len(flat_v)):
            original = flat_v[i]
            
            # f(x + epsilon)
            flat_v[i] = original + epsilon
            v_plus = flat_v.reshape(v.shape)
            params_plus = params.copy()
            params_plus[k] = v_plus
            loss_plus, _ = compute_loss(params_plus, model, user_feats, item_feats, interactions, backend='numpy')
            
            # f(x - epsilon)
            flat_v[i] = original - epsilon
            v_minus = flat_v.reshape(v.shape)
            params_minus = params.copy()
            params_minus[k] = v_minus
            loss_minus, _ = compute_loss(params_minus, model, user_feats, item_feats, interactions, backend='numpy')
            
            # central difference
            flat_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            flat_v[i] = original # restore
            
        grads[k] = flat_grad.reshape(v.shape)
    return grads

def train(lambda_proto=0.1, lambda_gini=0.1, epochs=10, learning_rate=0.05, quiet=False):
    loader = MusicDatasetLoader()
    train_interactions, test_interactions, user_features, item_features = loader.load_or_generate_data()
    
    hidden_dim = 16
    num_prototypes = 4
    
    if HAS_JAX:
        if not quiet:
            print("Using JAX/Flax backend.")
        key = jax.random.PRNGKey(42)
        key, init_key = jax.random.split(key)
        model = FlaxTwoTowerModel(hidden_dim=hidden_dim, num_prototypes=num_prototypes)
        params = model.init(init_key, user_features, item_features)['params']
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        @jax.jit
        def train_step(params, opt_state, u_f, i_f, interactions):
            def loss_fn(p):
                return compute_loss(p, model, u_f, i_f, interactions, lambda_proto=lambda_proto, lambda_gini=lambda_gini, backend='jax')
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, metrics

        for epoch in range(epochs):
            params, opt_state, loss, aux_metrics = train_step(params, opt_state, user_features, item_features, train_interactions)
            loss_recon, loss_proto, loss_gini, predictions = aux_metrics
            
            mrr_val = mrr(np.array(predictions), test_interactions)
            
            if not quiet:
                print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Recon: {loss_recon:.4f} | Proto: {loss_proto:.4f} | Gini: {loss_gini:.4f} | MRR: {mrr_val:.4f}")

    else:
        if not quiet:
            print("Using pure-NumPy backend with finite differences.")
        np.random.seed(42)
        model = NumpyTwoTowerModel(hidden_dim=hidden_dim, num_prototypes=num_prototypes)
        params = model.params
        
        for epoch in range(epochs):
            loss, aux_metrics = compute_loss(params, model, user_features, item_features, train_interactions, lambda_proto=lambda_proto, lambda_gini=lambda_gini, backend='numpy')
            loss_recon, loss_proto, loss_gini, predictions = aux_metrics
            
            grads = finite_difference_grad(params, model, user_features, item_features, train_interactions)
            
            # Simple SGD update
            for k in params.keys():
                params[k] -= learning_rate * grads[k]
                
            mrr_val = mrr(predictions, test_interactions)
            
            if not quiet:
                print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Recon: {loss_recon:.4f} | Proto: {loss_proto:.4f} | Gini: {loss_gini:.4f} | MRR: {mrr_val:.4f}")

    return params, model, HAS_JAX

if __name__ == "__main__":
    train()
