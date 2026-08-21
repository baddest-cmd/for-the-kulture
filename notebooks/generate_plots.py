"""
Module Name and Role: generate_plots.py - Component of notebooks/generate_plots.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

# Set style
sns.set_style('whitegrid')

# 1. Grouped horizontal bar chart for General vs. Local discovery shift
# Data prep
general = df['discovery_method'].value_counts(normalize=True) * 100
local = df['local_discovery_method'].value_counts(normalize=True) * 100

shifts = pd.DataFrame({
    'General Discovery': general,
    'Local Discovery': local
}).fillna(0).reset_index()
shifts.rename(columns={'index': 'Discovery Method'}, inplace=True)

# Sort by General Discovery for visual coherence
shifts = shifts.sort_values(by='General Discovery', ascending=True)

# Melt for seaborn grouped barplot
shifts_melted = pd.melt(shifts, id_vars='Discovery Method', var_name='Context', value_name='Percentage (%)')

# Categorical for correct ordering
shifts_melted['Discovery Method'] = pd.Categorical(
    shifts_melted['Discovery Method'],
    categories=shifts['Discovery Method'],
    ordered=True
)

plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x='Percentage (%)',
    y='Discovery Method',
    hue='Context',
    data=shifts_melted,
    palette=['#1f77b4', '#ff7f0e']
)
plt.title('Shift in Discovery Methods: General vs. Local (Algorithmic Localisation Bias)', fontsize=14, pad=15)
plt.xlabel('Percentage of Listeners (%)', fontsize=12)
plt.ylabel('')
plt.legend(title='Context', loc='lower right')
plt.tight_layout()
plt.savefig('../reports/discovery_shift_gap.png', dpi=300)
plt.close()

# 2. Boxplot of Algorithmic Ratings grouped by Age band
plt.figure(figsize=(8, 5))
# Sort age bands for logical ordering
age_order = sorted([x for x in df['age_band'].unique() if isinstance(x, str)])
sns.boxplot(
    x='rec_system_rating',
    y='age_band',
    data=df,
    order=age_order,
    palette='Set2',
    hue='age_band',
    legend=False
)
plt.title('Algorithmic Ratings by Age Band', fontsize=14, pad=15)
plt.xlabel('Algorithmic Rating (1-7)', fontsize=12)
plt.ylabel('Age Band', fontsize=12)
plt.tight_layout()
plt.savefig('../reports/rating_vs_constraints.png', dpi=300)
plt.close()

print("Publication-ready plots generated and saved successfully!")
