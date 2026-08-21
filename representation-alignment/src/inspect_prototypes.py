"""
Module Name and Role: inspect_prototypes.py - Component of representation-alignment/src/inspect_prototypes.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np
import sys
import os

# Add the current directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train

def main():
    print("Training model to extract prototypes...")
    params, model, has_jax = train()
    
    print("\n" + "="*50)
    print("Extracting prototypes...")
    if has_jax:
        import jax.numpy as jnp
        # params is a frozen dict, prototypes are in params['prototypes']
        prototypes = params['prototypes']
        prototypes = prototypes / jnp.linalg.norm(prototypes, axis=1, keepdims=True)
        prototypes = np.array(prototypes)
    else:
        prototypes = model.get_prototypes(params)
    
    # Compute pairwise cosine similarity
    # Since they are on the unit sphere, dot product is cosine similarity
    similarity_matrix = np.dot(prototypes, prototypes.T)
    
    # Hardcoded mapping to South African subgenres
    genres = ["Amapiano", "Gqom", "Lekompo", "Maskandi"]
    
    # Print ASCII similarity matrix
    print("\nSUBGENRE PROTOTYPE COSINE SIMILARITY MATRIX")
    print("="*50)
    
    # Header
    header = f"{'':>12}" + "".join([f"{g:>12}" for g in genres])
    print(header)
    print("-" * len(header))
    
    similarities = []
    
    for i, g1 in enumerate(genres):
        row = f"{g1:>12} |"
        for j, g2 in enumerate(genres):
            sim = similarity_matrix[i, j]
            row += f"{sim:11.4f} "
            if i < j:
                similarities.append(sim)
        print(row)
        
    print("-" * len(header))
    
    sims = np.array(similarities)
    min_sim = np.min(sims)
    max_sim = np.max(sims)
    avg_sim = np.mean(sims)
    
    print("\nSTATISTICS (Off-diagonal):")
    print(f"Minimum Similarity: {min_sim:.4f}")
    print(f"Maximum Similarity: {max_sim:.4f}")
    print(f"Average Similarity: {avg_sim:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
