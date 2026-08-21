"""
Module Name and Role: models.py - Demonstration script for the Paradigm A SCRUF-D recommender
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np

from kulture.postprocessing.scruf_d import ScrufdRecommender


def main():
    print("=== Paradigm A: 3-Agent SCRUF-D Recommendation Committee ===")
    
    # Initialize recommender
    recommender = ScrufdRecommender(
        w_exploit=0.5,
        w_arbiter=0.3,
        w_preserve=0.2,
        min_cultural_quota=3,
        k=5
    )
    
    # Mock items: 20 candidate songs across subgenres
    genres = [
        "Pop", "Pop", "HipHop", "HipHop", "Amapiano", 
        "Amapiano", "Gqom", "Gqom", "Lekompo", "Maskandi",
        "Pop", "EDM", "Afrobeats", "Afrobeats", "SA House",
        "SA House", "Amapiano", "Pop", "HipHop", "Maskandi"
    ]
    popularities = np.linspace(0.1, 1.0, len(genres))
    
    # Mock user 1: High context tax (tau_c = 0.6), baseline model prefers generic pop
    np.random.seed(42)
    predictions = np.random.uniform(0.1, 0.9, len(genres))
    
    result = recommender.recommend(
        user_id=101,
        predictions=predictions,
        context_tax=0.6,
        item_genres=genres,
        item_popularities=popularities
    )
    
    print(f"\nUser ID: {result['user_id']}")
    print(f"Context Tax (tau_c): {result['context_tax']:+.2f}")
    print(f"Selected Slate Indices: {result['slate_indices']}")
    print(f"Selected Slate Genres: {result['slate_genres']}")
    print(f"Cultural Items in Slate: {result['cultural_item_count']} / {recommender.k}")
    print(f"Cultural Quota Met (>= {recommender.min_cultural_quota}): {result['cultural_quota_met']}")
    print(f"Mean Predicted Engagement: {result['mean_slate_predicted_utility']:.4f}")


if __name__ == "__main__":
    main()
