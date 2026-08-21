"""
Module Name and Role: ablation_study.py - Component of representation-alignment/src/ablation_study.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np
import pandas as pd
import sys
import os
import contextlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import compute_loss, HAS_JAX
from utils import MusicDatasetLoader
from metrics import mrr, hr_at_k

if HAS_JAX:
    from models import TwoTowerModel as FlaxTwoTowerModel
else:
    from models import TwoTowerModel as NumpyTwoTowerModel

def local_finite_difference_grad(params, model, user_feats, item_feats, interactions, lp, lg, epsilon=1e-4):
    grads = {}
    for k, v in params.items():
        flat_v = v.flatten()
        flat_grad = np.zeros_like(flat_v)
        
        for i in range(len(flat_v)):
            original = flat_v[i]
            
            flat_v[i] = original + epsilon
            v_plus = flat_v.reshape(v.shape)
            params_plus = params.copy()
            params_plus[k] = v_plus
            loss_plus, _ = compute_loss(params_plus, model, user_feats, item_feats, interactions, lp, lg, backend='numpy')
            
            flat_v[i] = original - epsilon
            v_minus = flat_v.reshape(v.shape)
            params_minus = params.copy()
            params_minus[k] = v_minus
            loss_minus, _ = compute_loss(params_minus, model, user_feats, item_feats, interactions, lp, lg, backend='numpy')
            
            flat_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            flat_v[i] = original 
            
        grads[k] = flat_grad.reshape(v.shape)
    return grads

def evaluate_config(lambda_proto, lambda_gini):
    # Hide prints from load_or_generate_data
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        loader = MusicDatasetLoader()
        train_interactions, test_interactions, user_features, item_features = loader.load_or_generate_data()
    
    hidden_dim = 16
    num_prototypes = 4
    epochs = 10
    learning_rate = 0.05
    
    if HAS_JAX:
        import jax
        import optax
        key = jax.random.PRNGKey(42)
        model = FlaxTwoTowerModel(hidden_dim=hidden_dim, num_prototypes=num_prototypes)
        params = model.init(key, user_features, item_features)['params']
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        @jax.jit
        def train_step(params, opt_state, u_f, i_f, interactions):
            def loss_fn(p):
                return compute_loss(p, model, u_f, i_f, interactions, lambda_proto, lambda_gini, backend='jax')
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, metrics

        for epoch in range(epochs):
            params, opt_state, loss, aux_metrics = train_step(params, opt_state, user_features, item_features, train_interactions)
            loss_recon, loss_proto, loss_gini, predictions = aux_metrics
        
        predictions = np.array(predictions)
        mrr_val = mrr(predictions, test_interactions)
        hr_val = hr_at_k(predictions, test_interactions, k=5)
        
        return float(loss_recon), float(loss_gini), float(mrr_val), float(hr_val)
    else:
        model = NumpyTwoTowerModel(hidden_dim=hidden_dim, num_prototypes=num_prototypes)
        params = model.params
        
        for epoch in range(epochs):
            grads = local_finite_difference_grad(params, model, user_features, item_features, train_interactions, lambda_proto, lambda_gini)
            for k in params.keys():
                params[k] -= learning_rate * grads[k]
                
        loss, aux_metrics = compute_loss(params, model, user_features, item_features, train_interactions, lambda_proto, lambda_gini, backend='numpy')
        loss_recon, loss_proto, loss_gini, predictions = aux_metrics
        mrr_val = mrr(predictions, test_interactions)
        hr_val = hr_at_k(predictions, test_interactions, k=5)
        
        return float(loss_recon), float(loss_gini), float(mrr_val), float(hr_val)

def main():
    print("Starting Ablation Study on Hyperparameter Grid...\n")
    lambda_protos = [0.0, 0.1, 0.5, 1.0]
    lambda_ginis = [0.0, 0.1, 0.5, 1.0]
    
    results = []
    
    for lp in lambda_protos:
        for lg in lambda_ginis:
            print(f"Training config: lambda_proto={lp:.1f}, lambda_gini={lg:.1f} ...")
            recon, gini, mrr_v, hr_v = evaluate_config(lp, lg)
            results.append({
                'lambda_proto': lp,
                'lambda_gini': lg,
                'Recon_MSE': recon,
                'Gini_Exposure': gini,
                'MRR': mrr_v,
                'HR@5': hr_v
            })
            
    df = pd.DataFrame(results)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'ablation_results.csv'), index=False)
    
    print("\n# Ablation Study Results")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
        
    valid_configs = df[df['Gini_Exposure'] < 0.35]
    if len(valid_configs) > 0:
        optimal = valid_configs.loc[valid_configs['MRR'].idxmax()]
        print("\n### Pareto-Optimal Configuration")
        print("Found optimal hyperparameter pairing that maximizes MRR while keeping Gini < 0.35:")
        print(f"- **lambda_proto**: {optimal['lambda_proto']}")
        print(f"- **lambda_gini**: {optimal['lambda_gini']}")
        print(f"- **Gini Exposure**: {optimal['Gini_Exposure']:.4f}")
        print(f"- **MRR**: {optimal['MRR']:.4f}")
    else:
        print("\n### Pareto-Optimal Configuration")
        print("No configuration achieved Gini < 0.35.")

if __name__ == "__main__":
    main()
