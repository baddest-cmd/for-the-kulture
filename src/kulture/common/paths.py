"""
Module Name and Role: paths.py - Component of src/kulture/common/paths.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from pathlib import Path

# Resolve repository root by traversing up from this file:
# src/kulture/common/paths.py -> parents[3] is the repository root
REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = REPO_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

RAW_SURVEY_CSV = DATA_RAW_DIR / "fan_survey.csv"
CLEANED_SURVEY_CSV = DATA_PROCESSED_DIR / "fan_survey_cleaned.csv"
RAW_CLEANED_SURVEY_CSV = DATA_RAW_DIR / "fan_survey_cleaned.csv"

REPORTS_DIR = REPO_ROOT / "reports"
REP_ALIGN_DIR = REPO_ROOT / "representation-alignment"
REP_ALIGN_DATA_DIR = REP_ALIGN_DIR / "data" / "processed"


def get_dataset_path(preferred="processed") -> Path:
    """
    Returns the path to the cleaned survey CSV, falling back safely if one is missing.
    """
    if preferred == "processed" and CLEANED_SURVEY_CSV.exists():
        return CLEANED_SURVEY_CSV
    if RAW_CLEANED_SURVEY_CSV.exists():
        return RAW_CLEANED_SURVEY_CSV
    if CLEANED_SURVEY_CSV.exists():
        return CLEANED_SURVEY_CSV
    return RAW_SURVEY_CSV
