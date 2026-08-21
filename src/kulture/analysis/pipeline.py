"""
Module Name and Role: pipeline.py - Component of src/kulture/analysis/pipeline.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from ..common.paths import get_dataset_path, REPORTS_DIR


class CulturalAnalysisPipeline:
    """
    End-to-end analytical pipeline evaluating empirical survey data (N=152).
    """

    def __init__(self, data_path: Optional[Union[str, pd.DataFrame]] = None):
        if data_path is None:
            path = get_dataset_path()
            self.df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        elif isinstance(data_path, pd.DataFrame):
            self.df = data_path.copy()
        else:
            self.df = pd.read_csv(data_path)

    def run_inversion_shift_analysis(self) -> Dict[str, Any]:
        """
        Computes the empirical shift between general music discovery P(B|M)
        and localized music discovery P(B|M, C).
        """
        if self.df.empty or "discovery_method" not in self.df.columns or "local_discovery_method" not in self.df.columns:
            return {}

        general = self.df["discovery_method"].value_counts(normalize=True) * 100.0
        local = self.df["local_discovery_method"].value_counts(normalize=True) * 100.0

        shifts = pd.DataFrame({
            "general_pct": general,
            "local_pct": local
        }).fillna(0.0)
        shifts["shift_pp"] = shifts["local_pct"] - shifts["general_pct"]

        algo_key = "My streaming app's recommendations/algorithm"
        social_key = "Social media (TikTok, Instagram, etc.)"

        algo_shift = float(shifts.loc[algo_key, "shift_pp"]) if algo_key in shifts.index else 0.0
        social_shift = float(shifts.loc[social_key, "shift_pp"]) if social_key in shifts.index else 0.0

        return {
            "shifts_by_method": shifts.to_dict(orient="index"),
            "algo_reliance_shift_pp": algo_shift,
            "social_discovery_shift_pp": social_shift,
        }

    def run_non_parametric_tests(self) -> Dict[str, Any]:
        """
        Runs Kruskal-Wallis and Mann-Whitney U tests across behavioral segments.
        """
        if self.df.empty or "rec_system_rating" not in self.df.columns:
            return {}

        df_clean = self.df.dropna(subset=["rec_system_rating"]).copy()
        df_clean["rec_system_rating"] = pd.to_numeric(df_clean["rec_system_rating"], errors="coerce")
        df_clean = df_clean.dropna(subset=["rec_system_rating"])

        results = {}

        # 1. Kruskal-Wallis for Age Band
        if "age_band" in df_clean.columns:
            age_groups = [
                group["rec_system_rating"].values
                for _, group in df_clean[df_clean["age_band"] != "Unknown"].groupby("age_band")
                if len(group) > 0
            ]
            if len(age_groups) > 2:
                stat, p_val = kruskal(*age_groups)
                results["age_band_kruskal"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                }

        # 2. Mann-Whitney U for AI Feature Users vs Non-Users
        if "used_ai_features" in df_clean.columns:
            ai_yes = df_clean[df_clean["used_ai_features"] == "Yes"]["rec_system_rating"].values
            ai_no = df_clean[df_clean["used_ai_features"] == "No"]["rec_system_rating"].values
            if len(ai_yes) > 0 and len(ai_no) > 0:
                stat, p_val = mannwhitneyu(ai_yes, ai_no, alternative="two-sided")
                results["ai_adoption_mannwhitney"] = {
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05),
                    "median_yes": float(np.median(ai_yes)),
                    "median_no": float(np.median(ai_no)),
                }

        return results

    def run_lda_topic_modeling(self, n_topics: int = 3, top_words_per_topic: int = 5) -> Dict[str, Any]:
        """
        Extracts latent topics from user challenge feedback.
        """
        col = "challenge_following_local_artists"
        if self.df.empty or col not in self.df.columns:
            return {}

        texts = self.df[col].dropna().astype(str).tolist()
        texts = [t for t in texts if t.strip() and "no response" not in t.lower()]

        if len(texts) < 5:
            return {"topics": {}, "n_analyzed": len(texts)}

        vectorizer = CountVectorizer(stop_words="english", min_df=2, max_df=0.9)
        try:
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
            lda.fit(X)

            topics = {}
            for idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[:-top_words_per_topic - 1:-1]
                topics[f"Topic_{idx + 1}"] = [feature_names[i] for i in top_indices]

            return {
                "topics": topics,
                "n_analyzed": len(texts),
            }
        except ValueError:
            return {"topics": {}, "n_analyzed": len(texts)}

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Executes all analyses and returns a consolidated dictionary.
        """
        return {
            "inversion_shift": self.run_inversion_shift_analysis(),
            "non_parametric": self.run_non_parametric_tests(),
            "lda_topics": self.run_lda_topic_modeling(),
        }
