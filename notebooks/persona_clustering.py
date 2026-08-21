"""
Module Name and Role: persona_clustering.py - Component of notebooks/persona_clustering.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import numpy as np
from prince import MCA
from sklearn.cluster import KMeans
import json

df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

# Features for clustering
features = ['age_band', 'hours_streaming_weekly', 'discovery_method', 'primary_platforms', 'top_genres']
df_cluster = df[features].dropna()
df_cluster = df_cluster[(df_cluster != 'Unknown').all(axis=1)]
df_cluster = df_cluster[(df_cluster != 'No response').all(axis=1)]

# Simplify multi-selects by taking the first listed (primary) or counting the number
# Actually MCA can handle one-hot encoded multi-selects, but for simplicity let's extract the "primary" one
# or just use the whole string since MCA treats each unique string as a category. 
# But high cardinality strings (34 unique combos) dilute MCA. 
# Better: extract "Has Spotify", "Has Apple Music", "Has Amapiano", "Has Afrobeats"

df_cluster['has_spotify'] = df_cluster['primary_platforms'].str.contains('Spotify').astype(str)
df_cluster['has_apple'] = df_cluster['primary_platforms'].str.contains('Apple Music').astype(str)
df_cluster['has_youtube'] = df_cluster['primary_platforms'].str.contains('YouTube').astype(str)

df_cluster['has_amapiano'] = df_cluster['top_genres'].str.contains('Amapiano').astype(str)
df_cluster['has_gospel'] = df_cluster['top_genres'].str.contains('Gospel').astype(str)
df_cluster['has_afrobeats'] = df_cluster['top_genres'].str.contains('Afrobeats').astype(str)
df_cluster['has_hiphop'] = df_cluster['top_genres'].str.contains('Hip-Hop/Rap').astype(str)
df_cluster['has_sahouse'] = df_cluster['top_genres'].str.contains('SA House').astype(str)

# Drop original multi-selects
X = df_cluster.drop(columns=['primary_platforms', 'top_genres'])

# Run MCA
mca = MCA(n_components=3, n_iter=10, random_state=42)
mca.fit(X)
coords = mca.transform(X)

# Run KMeans on MCA coordinates
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_cluster['cluster'] = kmeans.fit_predict(coords)

results = {}
for i in range(n_clusters):
    cluster_data = df_cluster[df_cluster['cluster'] == i]
    size = len(cluster_data)
    
    # Calculate mode for categorical
    summary = {}
    for col in X.columns:
        summary[col] = cluster_data[col].mode().iloc[0]
        
    results[f"Persona_{i+1}"] = {
        'size': size,
        'percent': float(size / len(df_cluster) * 100),
        'attributes': summary
    }

with open('persona_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Persona Clustering complete.")
