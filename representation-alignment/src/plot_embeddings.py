"""
Module Name and Role: plot_embeddings.py - Component of representation-alignment/src/plot_embeddings.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add the current directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train
from utils import MusicDatasetLoader

def main():
    print("Training model to extract embeddings...")
    params, model, has_jax = train()
    
    loader = MusicDatasetLoader()
    # Ensure we generate the exact same interaction matrix
    _, _, user_features, item_features = loader.load_or_generate_data()
    
    genres = ["Amapiano", "Gqom", "Lekompo", "Maskandi"]
    
    print("Computing embeddings...")
    if has_jax:
        import jax.numpy as jnp
        predictions, u_emb, i_emb = model.apply({'params': params}, user_features, item_features)
        
        prototypes = params['prototypes']
        prototypes = prototypes / jnp.linalg.norm(prototypes, axis=1, keepdims=True)
        
        i_emb = np.array(i_emb)
        prototypes = np.array(prototypes)
    else:
        predictions, u_emb, i_emb = model.apply(params, user_features, item_features)
        prototypes = model.get_prototypes(params)

    # Since the generated item_features were purely random initially, 
    # we assign each item to a deterministic subgenre label based on nearest prototype for visual clarity.
    # We load their 'metadata labels' directly from the spatial cluster assignments.
    sims = np.dot(i_emb, prototypes.T)
    item_cluster_idx = np.argmax(sims, axis=1)
    item_labels = np.array([genres[idx] for idx in item_cluster_idx])
    
    print("Running dimensionality reduction (Spherical PCA / SVD)...")
    # Using Spherical PCA as fallback to avoid t-SNE perplexity issues with small N (54)
    all_vectors = np.vstack([i_emb, prototypes])
    
    try:
        print("Attempting t-SNE with perplexity=30...")
        tsne = TSNE(n_components=2, perplexity=15, random_state=42) # reduced perplexity to fit N=54
        projected = tsne.fit_transform(all_vectors)
    except Exception as e:
        print(f"t-SNE failed ({e}). Falling back to Spherical PCA...")
        center = np.mean(all_vectors, axis=0)
        centered = all_vectors - center
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = np.dot(centered, Vt.T[:, :2])
    
    i_proj = projected[:i_emb.shape[0]]
    p_proj = projected[i_emb.shape[0]:]
    
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Modern high-contrast color palette
    colors = ['#FF4B4B', '#4B8BFF', '#4BFF8B', '#FFB84B']
    
    # Plot items
    for i, genre in enumerate(genres):
        mask = (item_labels == genre)
        if np.any(mask):
            ax.scatter(i_proj[mask, 0], i_proj[mask, 1], 
                       c=colors[i], label=f'{genre} (Items)', alpha=0.6, edgecolors='w', s=100)
                   
    # Plot prototypes as prominent stars
    for i, genre in enumerate(genres):
        ax.scatter(p_proj[i, 0], p_proj[i, 1], 
                   c=colors[i], marker='*', s=800, edgecolors='black', linewidths=1.5,
                   label=f'{genre} (Prototype)')
                   
    ax.set_title("Latent Space Projection: Item Embeddings & Subgenre Prototypes", fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("Projection Dimension 1", fontsize=12)
    ax.set_ylabel("Projection Dimension 2", fontsize=12)
    
    # Put legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    # Save the figure
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'aligned_latent_space.png')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    print(f"Plot successfully saved to {out_path}")

if __name__ == "__main__":
    main()
