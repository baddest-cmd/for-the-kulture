"""
Module Name and Role: analysis.py - Component of src/analysis.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

from kulture.analysis.pipeline import CulturalAnalysisPipeline


def main():
    print("=== Paradigm A: Sociotechnical & Non-Parametric Analysis Pipeline ===")
    pipeline = CulturalAnalysisPipeline()
    results = pipeline.run_full_pipeline()
    
    # 1. Inversion Shift
    shift = results.get("inversion_shift", {})
    print("\n--- 1. The Inversion Shift [P(B|M) -> P(B|M, C)] ---")
    print(f"Drop in Algorithmic Reliance: {shift.get('algo_reliance_shift_pp', 0.0):+.2f} percentage points")
    print(f"Surge in Social Media Reliance: {shift.get('social_discovery_shift_pp', 0.0):+.2f} percentage points")
    
    # 2. Non-parametric Tests
    np_tests = results.get("non_parametric", {})
    print("\n--- 2. Non-Parametric Hypothesis Tests ---")
    for test_name, test_data in np_tests.items():
        print(f"• {test_name}: stat={test_data.get('statistic', 0.0):.3f}, p={test_data.get('p_value', 1.0):.4e}, significant={test_data.get('significant', False)}")
        
    # 3. Topic Modeling
    topics = results.get("lda_topics", {}).get("topics", {})
    print("\n--- 3. Latent Dirichlet Allocation (Challenge Topics) ---")
    for t_name, words in topics.items():
        print(f"• {t_name}: {', '.join(words)}")


if __name__ == "__main__":
    main()
