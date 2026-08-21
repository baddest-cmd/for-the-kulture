"""
Module Name and Role: __init__.py - Component of src/kulture/common/__init__.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from .paths import (
    REPO_ROOT,
    DATA_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    CLEANED_SURVEY_CSV,
    RAW_SURVEY_CSV,
    REPORTS_DIR,
    get_dataset_path,
)

__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "CLEANED_SURVEY_CSV",
    "RAW_SURVEY_CSV",
    "REPORTS_DIR",
    "get_dataset_path",
]
