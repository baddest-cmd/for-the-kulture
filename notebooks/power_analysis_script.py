"""
Module Name and Role: power_analysis_script.py - Component of notebooks/power_analysis_script.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import numpy as np
from statsmodels.stats.power import TTestIndPower, tt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest, power_proportions_2indep

df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

# 1. Bounding Pilot Power for N=152
# Assuming two groups (e.g., A/B test) with equal split (76 per group)
n_total = 152
n_per_group = n_total / 2
alpha = 0.05
power = 0.80

# Minimum Detectable Effect (MDE) for a two-sample t-test (Cohen's d)
ttest_power = TTestIndPower()
mde_d = ttest_power.solve_power(effect_size=None, nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0, alternative='two-sided')

# For proportions (e.g., Conversion Rate or 'Yes' vs 'No')
# MDE in proportion given a baseline (e.g., 50%)
# solve_power for proportions is a bit tricky, but we can compute it for a few baselines.
def get_mde_prop(baseline_prop, n_per_group, alpha, power):
    # Search for effect size
    from scipy import optimize
    def f(effect):
        return tt_ind_solve_power(effect_size=effect, nobs1=n_per_group, alpha=alpha, ratio=1.0, alternative='two-sided') - power
    try:
        eff = optimize.brentq(f, 0.001, 2.0)
        # proportion_effectsize(prop1, prop2) -> we want to find prop2 that gives this effect size
        # prop1 is baseline_prop.
        def f_prop(p2):
            return proportion_effectsize(baseline_prop, p2) - eff
        
        # p2 must be > baseline_prop and <= 1.0
        p2_high = optimize.brentq(f_prop, baseline_prop + 1e-5, 0.9999)
        return p2_high - baseline_prop
    except:
        return None

mde_prop_50 = get_mde_prop(0.5, n_per_group, alpha, power)

print(f"=== 1. BOUNDING PILOT POWER (N={n_total}) ===")
print(f"With {n_total} total participants ({int(n_per_group)} per group), the Minimum Detectable Effect (MDE) at 80% power and alpha=0.05 is:")
print(f"- Continuous Outcome (Cohen's d): {mde_d:.3f} standard deviations")
if mde_prop_50:
    print(f"- Binary Outcome (Baseline 50%): Absolute change of ~{mde_prop_50*100:.1f} percentage points")


# 2. Power per region/province for an AB test
# Let's see the province distribution
print("\n=== 2. UPSTREAM RESEARCH: POWER PER PROVINCE ===")
province_counts = df['province'].value_counts()
print("Current Province Distribution:")
print(province_counts)

# Let's say we want to detect a "Medium" effect size (Cohen's d = 0.5) per province
# or a "Small" effect size (Cohen's d = 0.2)
n_medium = ttest_power.solve_power(effect_size=0.5, nobs1=None, alpha=alpha, power=power, ratio=1.0, alternative='two-sided')
n_small = ttest_power.solve_power(effect_size=0.2, nobs1=None, alpha=alpha, power=power, ratio=1.0, alternative='two-sided')

print(f"\nTo run an A/B test PER PROVINCE and detect a MEDIUM effect (d=0.5):")
print(f"Required N per group: {np.ceil(n_medium)}")
print(f"Required Total N per province: {np.ceil(n_medium)*2}")

print(f"\nTo run an A/B test PER PROVINCE and detect a SMALL effect (d=0.2):")
print(f"Required N per group: {np.ceil(n_small)}")
print(f"Required Total N per province: {np.ceil(n_small)*2}")

print("\nFeasibility based on current pilot N=152:")
for province, count in province_counts.items():
    if count >= np.ceil(n_medium)*2:
        print(f"- {province}: {count} users -> Enough for medium effect per-province test")
    else:
        print(f"- {province}: {count} users -> Too small for medium effect. Needs {(np.ceil(n_medium)*2) - count} more users.")
