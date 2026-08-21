"""
Module Name and Role: __init__.py - Component of src/kulture/alignment/__init__.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from .metrics import differentiable_gini, mrr, hr_at_k
from .models import TwoTowerModel, FlaxTwoTowerModel, NumpyTwoTowerModel
from .utils import MusicDatasetLoader
from .train import train, compute_loss

__all__ = [
    "differentiable_gini",
    "mrr",
    "hr_at_k",
    "TwoTowerModel",
    "FlaxTwoTowerModel",
    "NumpyTwoTowerModel",
    "MusicDatasetLoader",
    "train",
    "compute_loss",
]
