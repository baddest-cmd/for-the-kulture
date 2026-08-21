"""
Module Name and Role: print_rules.py - Component of notebooks/print_rules.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import json

with open('market_basket_results.json', 'r') as f:
    data = json.load(f)

print("TOP GENRE-PLATFORM RULES:")
for rule in data.get('genre_platform_rules', []):
    print(f"Rule: {rule['rule']}")
    print(f"Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}\n")
    
print("TOP PLATFORM RULES:")
for rule in data.get('platform_rules', [])[:5]:
    print(f"Rule: {rule['rule']}")
    print(f"Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}\n")
