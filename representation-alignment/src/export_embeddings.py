"""
Module Name and Role: export_embeddings.py - Component of representation-alignment/src/export_embeddings.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the current directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train
from utils import MusicDatasetLoader

def main():
    print("Training model to extract final embeddings...")
    params, model, has_jax = train()
    
    loader = MusicDatasetLoader()
    # We need the metadata dataframe explicitly
    metadata_df = loader._generate_synthetic_data()
    # Ensure interactions align correctly with the generated random seeds
    _, _, user_features, item_features = loader._preprocess(metadata_df)
    
    genres = ["Amapiano", "Gqom", "Lekompo", "Maskandi"]
    
    print("Running final forward pass...")
    if has_jax:
        import jax.numpy as jnp
        predictions, u_emb, i_emb = model.apply({'params': params}, user_features, item_features)
        prototypes = params['prototypes']
        prototypes = prototypes / jnp.linalg.norm(prototypes, axis=1, keepdims=True)
        u_emb = np.array(u_emb)
        i_emb = np.array(i_emb)
        prototypes = np.array(prototypes)
    else:
        predictions, u_emb, i_emb = model.apply(params, user_features, item_features)
        prototypes = model.get_prototypes(params)

    # Determine item cluster labels
    sims = np.dot(i_emb, prototypes.T)
    item_cluster_idx = np.argmax(sims, axis=1)
    item_genres = [genres[idx] for idx in item_cluster_idx]
    
    hidden_dim = u_emb.shape[1]
    emb_cols = [f"emb_{i}" for i in range(hidden_dim)]
    
    print("Combining representations and metadata...")
    # Process User Data
    user_data = metadata_df.copy()
    user_data['entity_type'] = 'user'
    user_data['entity_id'] = user_data['user_id'].apply(lambda x: f"U_{x}")
    # Calculate Cultural Context Tax: Gap between global satisfaction and local discovery
    user_data['cultural_context_tax'] = user_data['satisfaction'] - user_data['discovery_success']
    
    user_embs_df = pd.DataFrame(u_emb, columns=emb_cols)
    user_df = pd.concat([user_data, user_embs_df], axis=1)
    user_df = user_df.rename(columns={'preferred_genre': 'genre'})
    
    # Process Item Data
    n_items = i_emb.shape[0]
    item_df = pd.DataFrame({
        'entity_id': [f"I_{i}" for i in range(n_items)],
        'entity_type': 'item',
        'genre': item_genres,
        'platform': 'N/A', # Items are cross-platform by default
        'discovery_success': np.nan,
        'satisfaction': np.nan,
        'cultural_context_tax': np.nan
    })
    item_embs_df = pd.DataFrame(i_emb, columns=emb_cols)
    item_df = pd.concat([item_df, item_embs_df], axis=1)
    
    # Combine Both
    final_df = pd.concat([user_df, item_df], ignore_index=True)
    
    # Clean up and reorder columns
    col_order = ['entity_id', 'entity_type', 'genre', 'platform', 
                 'discovery_success', 'satisfaction', 'cultural_context_tax'] + emb_cols
    final_df = final_df[col_order]
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'aligned_representations.csv')
    
    final_df.to_csv(out_path, index=False)
    
    print(f"\nExport complete! File saved to {out_path}")
    print(f"File Shape: {final_df.shape}")
    print("\nPreview of first 3 rows:")
    display_cols = ['entity_id', 'entity_type', 'genre', 'cultural_context_tax', 'emb_0', 'emb_1', 'emb_2']
    print(final_df[display_cols].head(3).to_string())

if __name__ == "__main__":
    main()
