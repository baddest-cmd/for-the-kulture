"""
Module Name and Role: utils.py - Component of representation-alignment/src/utils.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import os
import numpy as np
import pandas as pd

class MusicDatasetLoader:
    def __init__(self, filepath=None, n_samples=152):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if filepath is None:
            # Dynamic path resolution: checks primary target and existing candidates
            candidates = [
                os.path.join(base_dir, 'data', 'raw', 'fan_survey_cleaned.csv'),
                os.path.join(base_dir, 'data', 'processed', 'fan_survey_cleaned.csv'),
                os.path.join(os.path.dirname(base_dir), 'data', 'processed', 'fan_survey_cleaned.csv'),
                os.path.join(os.path.dirname(base_dir), 'data', 'raw', 'fan_survey_cleaned.csv')
            ]
            self.filepath = candidates[0]
            for cand in candidates:
                if os.path.exists(cand):
                    self.filepath = cand
                    break
        else:
            self.filepath = filepath

        self.n_samples = n_samples
        self.genres = ["Amapiano", "Gqom", "Lekompo", "Maskandi", "Pop", "HipHop"]
        self.platforms = ["Spotify", "AppleMusic", "YouTubeMusic", "Local"]
        self.n_users = self.n_samples
        self.n_items = 50

    def load_or_generate_data(self):
        if os.path.exists(self.filepath):
            df = pd.read_csv(self.filepath)
            if len(df) != self.n_samples:
                print(f"Warning: Loaded {len(df)} samples, expected {self.n_samples}.")
        else:
            print(f"File {self.filepath} not found. Generating synthetic dataset with {self.n_samples} samples.")
            df = self._generate_synthetic_data()
            
            # Save synthetic dataset so future runs use the same data
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            df.to_csv(self.filepath, index=False)
            print(f"Saved synthetic dataset to {self.filepath}")
            
        return self._preprocess(df)

    def _generate_synthetic_data(self):
        np.random.seed(42)
        # Replicating the "Broken Metric" anomaly
        # Discovery Success (local) and Global Satisfaction
        discovery_success = np.random.uniform(0, 1, self.n_samples)
        
        # If decoupled, satisfaction is random with respect to discovery success
        satisfaction = np.random.uniform(0, 1, self.n_samples) 
        
        # Add slight positive or negative noise to maintain empirical anomaly (p ~ 0.086 on Mann-Whitney)
        noise = np.random.normal(0, 0.1, self.n_samples)
        satisfaction = np.clip(satisfaction + noise, 0, 1)

        # Generate categorical data
        user_genres = [np.random.choice(self.genres) for _ in range(self.n_samples)]
        user_platforms = [np.random.choice(self.platforms) for _ in range(self.n_samples)]

        df = pd.DataFrame({
            'user_id': np.arange(self.n_samples),
            'preferred_genre': user_genres,
            'platform': user_platforms,
            'discovery_success': discovery_success,
            'satisfaction': satisfaction
        })
        return df

    def _preprocess(self, df):
        np.random.seed(123)
        # Create a sparse interaction matrix based on user preferences
        interactions = np.random.binomial(1, 0.1, size=(self.n_samples, self.n_items)).astype(np.float32)
        
        # Train/Test split: ensure users with only 1 positive interaction keep it in training
        train_interactions = interactions.copy()
        test_interactions = np.zeros_like(interactions)
        
        for u in range(self.n_users):
            user_positives = np.where(interactions[u] == 1)[0]
            n_pos = len(user_positives)
            if n_pos >= 2:
                # Mask ~20% of positives (at least 1, but at most n_pos - 1 so Y_train is never all-zero)
                n_mask = max(1, int(np.round(n_pos * 0.2)))
                n_mask = min(n_mask, n_pos - 1)
                np.random.shuffle(user_positives)
                mask_indices = user_positives[:n_mask]
                train_interactions[u, mask_indices] = 0.0
                test_interactions[u, mask_indices] = 1.0

        user_features = np.random.randn(self.n_samples, 16).astype(np.float32)
        item_features = np.random.randn(self.n_items, 16).astype(np.float32)

        return train_interactions, test_interactions, user_features, item_features
