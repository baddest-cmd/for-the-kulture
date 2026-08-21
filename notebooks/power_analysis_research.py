"""
Module Name and Role: power_analysis_research.py - Component of notebooks/power_analysis_research.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import numpy as np
from statsmodels.stats.power import FTestAnovaPower
from statsmodels.stats.proportion import power_proportions_2indep

alpha = 0.05
n_total = 152

power_algo_diff = power_proportions_2indep(0.0724, prop2=0.30, nobs1=n_total, ratio=1.0, alpha=alpha, alternative='two-sided')
power_social_diff = power_proportions_2indep(0.0461, prop2=0.30, nobs1=n_total, ratio=1.0, alpha=alpha, alternative='two-sided')

# Get the actual float value, statsmodels sometimes returns a HolderTuple
try:
    pa = float(power_algo_diff)
except TypeError:
    pa = power_algo_diff.power
try:
    ps = float(power_social_diff)
except TypeError:
    ps = power_social_diff.power

print("=== POWER FOR EMPIRICAL RESEARCH DIRECTION ===")
print(f"Power to detect a 7.24% shift in proportions (assuming ~30% baseline) with N=152 (two indep groups): {pa:.2%}")
print(f"Power to detect a 4.61% shift in proportions (assuming ~30% baseline) with N=152 (two indep groups): {ps:.2%}")

anova_power = FTestAnovaPower()
power_anova = anova_power.solve_power(effect_size=0.25, nobs=n_total, k_groups=4, alpha=alpha)
print(f"Power for ANOVA across 4 age bands detecting a moderate effect (f=0.25): {power_anova:.2%}")

power_anova_small = anova_power.solve_power(effect_size=0.10, nobs=n_total, k_groups=4, alpha=alpha)
print(f"Power for ANOVA across 4 age bands detecting a small effect (f=0.10): {power_anova_small:.2%}")
