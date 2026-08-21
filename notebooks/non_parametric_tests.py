"""
Module Name and Role: non_parametric_tests.py - Component of notebooks/non_parametric_tests.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import json

df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

# Drop rows where rating is missing
df = df.dropna(subset=['rec_system_rating'])

results = {}

def get_groups(col):
    df_clean = df.dropna(subset=[col])
    df_clean = df_clean[df_clean[col] != 'Unknown']
    
    groups = []
    labels = []
    for val in df_clean[col].unique():
        ratings = df_clean[df_clean[col] == val]['rec_system_rating'].values
        if len(ratings) > 0:
            groups.append(ratings)
            labels.append(val)
    return groups, labels

# 1. Kruskal-Wallis for Age Band (3+ groups)
groups_age, labels_age = get_groups('age_band')
if len(groups_age) > 2:
    stat, p = kruskal(*groups_age)
    results['age_band_kruskal'] = {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': bool(p < 0.05),
        'groups': labels_age
    }

# 2. Kruskal-Wallis for Discovery Method (3+ groups)
groups_disc, labels_disc = get_groups('discovery_method')
if len(groups_disc) > 2:
    stat, p = kruskal(*groups_disc)
    results['discovery_method_kruskal'] = {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': bool(p < 0.05),
        'groups': labels_disc
    }

# 3. Mann-Whitney U for Used AI Features (Yes vs No)
df_ai = df.dropna(subset=['used_ai_features'])
ai_yes = df_ai[df_ai['used_ai_features'] == 'Yes']['rec_system_rating'].values
ai_no = df_ai[df_ai['used_ai_features'] == 'No']['rec_system_rating'].values

if len(ai_yes) > 0 and len(ai_no) > 0:
    stat, p = mannwhitneyu(ai_yes, ai_no, alternative='two-sided')
    results['used_ai_features_mannwhitney'] = {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': bool(p < 0.05),
        'median_yes': float(pd.Series(ai_yes).median()),
        'median_no': float(pd.Series(ai_no).median())
    }

# 4. Mann-Whitney U for Discovered New Artist (Yes vs No)
df_new = df.dropna(subset=['discovered_new_artist_recently'])
new_yes = df_new[df_new['discovered_new_artist_recently'] == 'Yes']['rec_system_rating'].values
new_no = df_new[df_new['discovered_new_artist_recently'] == 'No']['rec_system_rating'].values

if len(new_yes) > 0 and len(new_no) > 0:
    stat, p = mannwhitneyu(new_yes, new_no, alternative='two-sided')
    results['discovered_new_artist_mannwhitney'] = {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': bool(p < 0.05),
        'median_yes': float(pd.Series(new_yes).median()),
        'median_no': float(pd.Series(new_no).median())
    }

with open('non_parametric_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Non-parametric tests complete.")
