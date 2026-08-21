"""
Module Name and Role: market_basket_analysis.py - Component of notebooks/market_basket_analysis.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import json

df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

results = {}

def get_rules(series, min_support=0.1, min_threshold=0.5):
    # Drop NAs and "Unknown"
    series = series.dropna()
    series = series[~series.str.contains('Unknown', case=False)]
    
    # Split by comma and strip whitespace
    transactions = [ [item.strip() for item in row.split(',')] for row in series ]
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apriori
    frequent_itemsets = apriori(df_trans, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        return []
        
    # Association Rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold, num_itemsets=len(frequent_itemsets))
    
    # Format output
    output_rules = []
    for idx, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        support = float(row['support'])
        confidence = float(row['confidence'])
        lift = float(row['lift'])
        
        # Only keep interesting rules (lift > 1 implies positive correlation)
        if lift > 1.0:
            output_rules.append({
                'rule': f"IF {antecedents} THEN {consequents}",
                'support': support,
                'confidence': confidence,
                'lift': lift
            })
            
    # Sort by lift, then confidence
    output_rules.sort(key=lambda x: (x['lift'], x['confidence']), reverse=True)
    return output_rules

# 1. Rules within Genres
results['genre_rules'] = get_rules(df['top_genres'], min_support=0.1, min_threshold=0.6)

# 2. Rules within Platforms
results['platform_rules'] = get_rules(df['primary_platforms'], min_support=0.1, min_threshold=0.6)

# 3. Cross-domain Rules (Genres AND Platforms)
# Combine both into a single transaction
combined = df.dropna(subset=['top_genres', 'primary_platforms'])
combined = combined[~combined['top_genres'].str.contains('Unknown') & ~combined['primary_platforms'].str.contains('Unknown')]

cross_transactions = []
for _, row in combined.iterrows():
    genres = [g.strip() + " (Genre)" for g in row['top_genres'].split(',')]
    platforms = [p.strip() + " (Platform)" for p in row['primary_platforms'].split(',')]
    cross_transactions.append(genres + platforms)
    
te_cross = TransactionEncoder()
te_ary_cross = te_cross.fit_transform(cross_transactions)
df_cross = pd.DataFrame(te_ary_cross, columns=te_cross.columns_)

freq_cross = apriori(df_cross, min_support=0.15, use_colnames=True)
rules_cross = association_rules(freq_cross, metric="confidence", min_threshold=0.6, num_itemsets=len(freq_cross))

cross_output = []
for idx, row in rules_cross.iterrows():
    antecedents = list(row['antecedents'])
    consequents = list(row['consequents'])
    
    # We are mainly interested in Rules where Genre -> Platform or vice versa
    ant_types = set([x.split('(')[-1] for x in antecedents])
    con_types = set([x.split('(')[-1] for x in consequents])
    
    if ant_types != con_types:
        support = float(row['support'])
        confidence = float(row['confidence'])
        lift = float(row['lift'])
        
        if lift > 1.1: # Strict lift filter for cross rules
            cross_output.append({
                'rule': f"IF {antecedents} THEN {consequents}",
                'support': support,
                'confidence': confidence,
                'lift': lift
            })

cross_output.sort(key=lambda x: (x['lift'], x['confidence']), reverse=True)
results['genre_platform_rules'] = cross_output[:10] # Top 10 rules

with open('market_basket_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Market Basket Analysis complete.")
