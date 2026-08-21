"""
Module Name and Role: scruf_d.py - Component of src/kulture/postprocessing/scruf_d.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class MyopicExploiterAgent:
    """Agent 1: Prioritizes immediate engagement probability y_hat."""
    def __init__(self, name: str = "MyopicExploiter"):
        self.name = name

    def score(self, predictions: np.ndarray) -> np.ndarray:
        """Raw prediction score."""
        return np.copy(predictions)


class CausalArbiterAgent:
    """
    Agent 2: Causal Alignment Framing Layer (CAFL).
    Penalizes mainstream over-indexing when user context tax tau_c is elevated.
    """
    def __init__(self, beta: float = 0.5, name: str = "CausalArbiter"):
        self.beta = beta
        self.name = name

    def score(
        self,
        predictions: np.ndarray,
        context_tax: float,
        item_popularities: np.ndarray
    ) -> np.ndarray:
        """
        Adjusted score:
        Score = predictions - beta * max(0, tau_c) * item_popularity
        """
        penalty = self.beta * max(0.0, context_tax) * item_popularities
        return predictions - penalty


class AdversarialPreserverAgent:
    """
    Agent 3: Enforces regional cultural preservation quotas (Q_g).
    Boosts underrepresented regional subgenres (e.g. Gqom, Lekompo, Maskandi).
    """
    def __init__(
        self,
        target_subgenres: Optional[List[str]] = None,
        quota_weight: float = 0.8,
        name: str = "AdversarialPreserver"
    ):
        self.target_subgenres = target_subgenres or ["Gqom", "Lekompo", "Maskandi", "SA House"]
        self.quota_weight = quota_weight
        self.name = name

    def score(self, item_genres: List[str]) -> np.ndarray:
        """Scores items based on protected cultural subgenre membership."""
        scores = np.zeros(len(item_genres), dtype=np.float32)
        for idx, genre in enumerate(item_genres):
            if genre in self.target_subgenres:
                scores[idx] = self.quota_weight
        return scores


class ScrufdRecommender:
    """
    Orchestrates the 3-agent SCRUF-D social choice negotiation committee.
    Produces fair recommendation slates at the decision boundary.
    """

    def __init__(
        self,
        w_exploit: float = 0.5,
        w_arbiter: float = 0.3,
        w_preserve: float = 0.2,
        min_cultural_quota: int = 2,
        k: int = 10
    ):
        # Normalize agent weights
        total_w = w_exploit + w_arbiter + w_preserve
        self.w_exploit = w_exploit / total_w
        self.w_arbiter = w_arbiter / total_w
        self.w_preserve = w_preserve / total_w
        
        self.min_cultural_quota = min_cultural_quota
        self.k = k

        self.agent_exploit = MyopicExploiterAgent()
        self.agent_arbiter = CausalArbiterAgent()
        self.agent_preserve = AdversarialPreserverAgent()

    def recommend(
        self,
        user_id: int,
        predictions: np.ndarray,
        context_tax: float,
        item_genres: List[str],
        item_popularities: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Negotiates recommendation slate for a single user.
        
        Returns:
            Dictionary with recommended indices, agent scores, and slate diversity metrics.
        """
        n_items = len(predictions)
        if item_popularities is None:
            item_popularities = np.linspace(0.1, 1.0, n_items)

        # 1. Agent scoring
        s_exploit = self.agent_exploit.score(predictions)
        s_arbiter = self.agent_arbiter.score(predictions, context_tax, item_popularities)
        s_preserve = self.agent_preserve.score(item_genres)

        # 2. Negotiated utility via weighted social welfare
        negotiated_scores = (
            self.w_exploit * s_exploit +
            self.w_arbiter * s_arbiter +
            self.w_preserve * s_preserve
        )

        # 3. Initial ranking
        sorted_indices = np.argsort(-negotiated_scores).tolist()
        
        # 4. Quota fulfillment check
        slate = sorted_indices[:self.k]
        target_genres = set(self.agent_preserve.target_subgenres)
        slate_cultural_count = sum(1 for idx in slate if item_genres[idx] in target_genres)

        # Enforce quota if shortfall occurs
        if slate_cultural_count < self.min_cultural_quota:
            needed = self.min_cultural_quota - slate_cultural_count
            remaining_candidates = [idx for idx in sorted_indices[self.k:] if item_genres[idx] in target_genres]
            
            # Replace bottom unpreserved items with top preserved items
            if remaining_candidates:
                swap_count = min(needed, len(remaining_candidates))
                # Find replaceable items from bottom of slate
                non_cultural_in_slate = [idx for idx in reversed(slate) if item_genres[idx] not in target_genres]
                for i in range(min(swap_count, len(non_cultural_in_slate))):
                    remove_idx = non_cultural_in_slate[i]
                    insert_idx = remaining_candidates[i]
                    slate.remove(remove_idx)
                    slate.append(insert_idx)

        slate_genres = [item_genres[idx] for idx in slate]
        final_cultural_count = sum(1 for g in slate_genres if g in target_genres)

        return {
            "user_id": user_id,
            "slate_indices": slate,
            "slate_genres": slate_genres,
            "cultural_item_count": final_cultural_count,
            "cultural_quota_met": bool(final_cultural_count >= self.min_cultural_quota),
            "mean_slate_predicted_utility": float(np.mean(predictions[slate])),
            "context_tax": float(context_tax),
        }
