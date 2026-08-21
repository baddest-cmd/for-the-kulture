"""
Module Name and Role: __init__.py - Component of src/kulture/analysis/__init__.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from .pipeline import CulturalAnalysisPipeline

__all__ = ["CulturalAnalysisPipeline"]
