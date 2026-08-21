"""
Module Name and Role: context_tax.py - Component of src/kulture/postprocessing/context_tax.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from ..common.paths import get_dataset_path


class ContextTaxCalculator:
    """
    Computes individual and cohort-level Context Tax (tau_c) scores.
    
    Formula:
        tau_c(u) = S_global(u) - D_local(u)
    where:
        S_global in [0, 1] is overall platform streaming satisfaction.
        D_local in [0, 1] is the native in-app local music discovery effectiveness.
    """

    def __init__(self, data_path: Optional[Union[str, pd.DataFrame]] = None):
        if data_path is None:
            path = get_dataset_path()
            self.df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        elif isinstance(data_path, pd.DataFrame):
            self.df = data_path.copy()
        else:
            self.df = pd.read_csv(data_path)

    @staticmethod
    def compute_individual_tax(
        satisfaction: Union[float, np.ndarray],
        discovery_success: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculates tau_c = S_global - D_local.
        Bounded in [-1.0, 1.0].
        """
        satisfaction = np.clip(satisfaction, 0.0, 1.0)
        discovery_success = np.clip(discovery_success, 0.0, 1.0)
        return satisfaction - discovery_success

    def compute_empirical_scores(self) -> pd.DataFrame:
        """
        Computes empirical Context Tax scores and platform-hopping indicators on survey data.
        """
        if self.df.empty:
            raise ValueError("Dataset is empty. Cannot compute empirical scores.")

        df = self.df.copy()

        # 1. Normalize rec_system_rating (1-7 scale) to [0, 1]
        if "rec_system_rating" in df.columns:
            ratings = pd.to_numeric(df["rec_system_rating"], errors="coerce")
            # Fill NAs with neutral rating 4.0
            ratings = ratings.fillna(4.0)
            df["s_global_norm"] = (ratings - 1.0) / 6.0
        else:
            df["s_global_norm"] = 0.5

        # 2. Derive D_local: 1.0 if discovered locally via in-app algorithm, 0.0 if forced to external channels
        in_app_algo_str = "My streaming app's recommendations/algorithm"
        if "local_discovery_method" in df.columns:
            df["d_local_native"] = df["local_discovery_method"].apply(
                lambda x: 1.0 if isinstance(x, str) and in_app_algo_str in x else 0.0
            )
        else:
            df["d_local_native"] = 0.5

        # 3. Calculate Context Tax
        df["context_tax"] = self.compute_individual_tax(
            df["s_global_norm"].values,
            df["d_local_native"].values
        )

        # 4. Platform-Hopping Index (Has major streaming app but discovers local music via TikTok/Social/IRL)
        if "discovery_method" in df.columns and "local_discovery_method" in df.columns:
            df["is_platform_hopper"] = (
                (df["discovery_method"] == in_app_algo_str) &
                (df["local_discovery_method"] != in_app_algo_str)
            ).astype(int)
        else:
            df["is_platform_hopper"] = 0

        return df

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Returns aggregate Context Tax and behavioral friction statistics.
        """
        df_scored = self.compute_empirical_scores()
        
        return {
            "mean_context_tax": float(df_scored["context_tax"].mean()),
            "median_context_tax": float(df_scored["context_tax"].median()),
            "std_context_tax": float(df_scored["context_tax"].std()),
            "platform_hopper_rate": float(df_scored["is_platform_hopper"].mean()),
            "high_tax_cohort_pct": float((df_scored["context_tax"] > 0.3).mean() * 100.0),
            "sample_size": len(df_scored),
        }
