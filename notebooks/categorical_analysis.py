"""
Module Name and Role: categorical_analysis.py - Component of notebooks/categorical_analysis.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import json

df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

# Clean "Unknown" or NA values for these specific tests to avoid spurious associations
def get_clean_crosstab(col1, col2):
    sub = df[[col1, col2]].dropna()
    sub = sub[(sub[col1] != 'Unknown') & (sub[col2] != 'Unknown') & (sub[col1] != 'No response') & (sub[col2] != 'No response')]
    return pd.crosstab(sub[col1], sub[col2])

tests = [
    ('age_band', 'discovery_method'),
    ('age_band', 'local_discovery_method'),
    ('age_band', 'used_ai_features'),
    ('province', 'discovery_method'),
    ('discovered_new_artist_recently', 'discovery_method')
]

results = {}

for col1, col2 in tests:
    ct = get_clean_crosstab(col1, col2)
    if ct.empty or ct.shape[0] < 2 or ct.shape[1] < 2:
        continue
    
    # Chi-Square Test
    chi2, p_chi2, dof, expected = chi2_contingency(ct)
    
    # Check if we have cells with expected frequency < 5
    low_counts = (expected < 5).sum()
    total_cells = expected.size
    pct_low = low_counts / total_cells
    
    warning = False
    if pct_low > 0.2:
        warning = True
        
    test_res = {
        'chi2_stat': float(chi2),
        'p_value': float(p_chi2),
        'dof': int(dof),
        'low_expected_count_warning': warning,
        'pct_cells_under_5': float(pct_low)
    }
    
    # If 2x2, we can run Fisher's Exact
    if ct.shape == (2, 2):
        oddsr, p_fisher = fisher_exact(ct)
        test_res['fisher_p_value'] = float(p_fisher)
        test_res['odds_ratio'] = float(oddsr)
        
    results[f"{col1}_vs_{col2}"] = test_res

with open('categorical_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Categorical analysis complete.")
