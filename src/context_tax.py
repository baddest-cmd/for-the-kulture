"""
Module Name and Role: context_tax.py - Component of src/context_tax.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from kulture.postprocessing.context_tax import ContextTaxCalculator


def main():
    print("=== Cultural Context Tax (tau_c) Calculation ===")
    calc = ContextTaxCalculator()
    stats = calc.get_summary_statistics()
    
    print(f"Sample Size (N): {stats['sample_size']}")
    print(f"Mean Context Tax (tau_c): {stats['mean_context_tax']:+.4f}")
    print(f"Median Context Tax: {stats['median_context_tax']:+.4f}")
    print(f"Standard Deviation: {stats['std_context_tax']:.4f}")
    print(f"Platform-Hopper Rate: {stats['platform_hopper_rate'] * 100:.1f}%")
    print(f"High-Tax Cohort (>0.30): {stats['high_tax_cohort_pct']:.1f}%")


if __name__ == "__main__":
    main()
