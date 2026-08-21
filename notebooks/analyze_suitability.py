"""
Module Name and Role: analyze_suitability.py - Component of notebooks/analyze_suitability.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import numpy as np
import json

df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

report = {}
report['n_rows'] = len(df)
report['n_cols'] = len(df.columns)
report['columns'] = list(df.columns)

# Analyze each column
col_stats = {}
for col in df.columns:
    stats = {}
    stats['dtype'] = str(df[col].dtype)
    stats['n_missing'] = int(df[col].isna().sum())
    stats['n_unique'] = int(df[col].nunique())
    if pd.api.types.is_numeric_dtype(df[col]):
        stats['mean'] = float(df[col].mean())
        stats['std'] = float(df[col].std())
        stats['min'] = float(df[col].min())
        stats['max'] = float(df[col].max())
    else:
        # Get top 5 value counts
        val_counts = df[col].value_counts().head(5).to_dict()
        stats['top_values'] = val_counts
    col_stats[col] = stats

report['column_stats'] = col_stats

with open('suitability_stats.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Data profiling complete. Wrote to suitability_stats.json")
