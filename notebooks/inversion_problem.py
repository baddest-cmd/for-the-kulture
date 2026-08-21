"""
Module Name and Role: inversion_problem.py - Component of notebooks/inversion_problem.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd

# Load the data (paths are relative to the project root, or we can use absolute. Better to run script from project root or adjust path)
df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

print("=== Inversion Problem Analysis ===\n")
print("Conceptual Framework:")
print("P(B|M)   -> General Discovery")
print("P(B|M,C) -> Local Discovery\n")

# Calculate discovery distributions
general = df['discovery_method'].value_counts(normalize=True) * 100
local = df['local_discovery_method'].value_counts(normalize=True) * 100

# Combine and calculate shifts
shifts = pd.DataFrame({
    'General_Discovery (%)': general,
    'Local_Discovery (%)': local
}).fillna(0)
shifts['Shift (percentage points)'] = shifts['Local_Discovery (%)'] - shifts['General_Discovery (%)']

print("--- Shift between General and Local Discovery Methods ---")
print(shifts.round(2).to_string())
print("\n")

# Highlight specific shifts
algo_name = "My streaming app's recommendations/algorithm"
social_name = "Social media (TikTok, Instagram, etc.)"

algo_shift = shifts.loc[algo_name, 'Shift (percentage points)']
social_shift = shifts.loc[social_name, 'Shift (percentage points)']

print("--- Key Findings ---")
print(f"Drop in Algorithmic Reliance: {algo_shift:+.2f} percentage points")
print(f"Surge in Social Media Usage: {social_shift:+.2f} percentage points")
print("\n")

print("--- Algorithmic_Rating cross-tabulated with Age band ---")
cross_tab = pd.crosstab(df['age_band'], df['rec_system_rating'], margins=True)
print(cross_tab.to_string())
