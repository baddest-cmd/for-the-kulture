"""
Module Name and Role: utils.py - Component of src/kulture/alignment/utils.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from ..common.paths import get_dataset_path, REP_ALIGN_DATA_DIR


class MusicDatasetLoader:
    """
    Handles robust data loading with fallback to empirical synthetic data.
    """
    def __init__(
        self,
        filepath: Optional[Union[str, Path]] = None,
        n_samples: int = 152,
        n_items: int = 50,
        feature_dim: int = 16,
        seed: int = 42
    ):
        if filepath is None:
            self.filepath = get_dataset_path()
        else:
            self.filepath = Path(filepath)

        self.n_samples = n_samples
        self.n_users = n_samples
        self.n_items = n_items
        self.feature_dim = feature_dim
        self.seed = seed
        self.genres = ["Amapiano", "Gqom", "Lekompo", "Maskandi", "Pop", "HipHop"]
        self.platforms = ["Spotify", "AppleMusic", "YouTubeMusic", "Local"]

    def load_or_generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (train_interactions, test_interactions, user_features, item_features).
        """
        if self.filepath.exists():
            df = pd.read_csv(self.filepath)
        else:
            df = self._generate_synthetic_data()
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.filepath, index=False)

        return self._preprocess(df)

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Replicates the empirical Inversion Problem / Broken Metric anomaly.
        """
        rng = np.random.default_rng(self.seed)
        discovery_success = rng.uniform(0.0, 1.0, self.n_samples)
        satisfaction = rng.uniform(0.0, 1.0, self.n_samples)
        noise = rng.normal(0.0, 0.1, self.n_samples)
        satisfaction = np.clip(satisfaction + noise, 0.0, 1.0)

        user_genres = [rng.choice(self.genres) for _ in range(self.n_samples)]
        user_platforms = [rng.choice(self.platforms) for _ in range(self.n_samples)]

        return pd.DataFrame({
            "user_id": np.arange(self.n_samples),
            "preferred_genre": user_genres,
            "platform": user_platforms,
            "discovery_success": discovery_success,
            "satisfaction": satisfaction
        })

    def _preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Builds interaction matrices with an invariant:
        Users with >= 2 positives keep at least 1 positive in both train and test.
        """
        rng = np.random.default_rng(self.seed + 81)
        interactions = rng.binomial(1, 0.1, size=(self.n_samples, self.n_items)).astype(np.float32)

        train_interactions = interactions.copy()
        test_interactions = np.zeros_like(interactions)

        for u in range(self.n_users):
            user_positives = np.where(interactions[u] == 1)[0]
            n_pos = len(user_positives)
            if n_pos >= 2:
                n_mask = max(1, int(np.round(n_pos * 0.2)))
                n_mask = min(n_mask, n_pos - 1)
                rng.shuffle(user_positives)
                mask_indices = user_positives[:n_mask]
                train_interactions[u, mask_indices] = 0.0
                test_interactions[u, mask_indices] = 1.0

        user_features = rng.normal(0.0, 1.0, (self.n_samples, self.feature_dim)).astype(np.float32)
        item_features = rng.normal(0.0, 1.0, (self.n_items, self.feature_dim)).astype(np.float32)

        return train_interactions, test_interactions, user_features, item_features
