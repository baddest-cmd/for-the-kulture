"""
Module Name and Role: qualitative_clustering.py - Component of src/qualitative_clustering.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import re

# Load the dataset
df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

# Combine the relevant text columns to evaluate them together
df['qualitative_text'] = df['rec_system_feedback'].fillna('') + " | " + df['challenge_following_local_artists'].fillna('')

# Define keywords for each theoretical bucket based on the literature review
buckets = {
    'Popularity Bias': [
        'popular', 'generic', 'grouped', 'common', 'neutral', 'mainstream', 'algorithm'
    ],
    'Contextual Constraints': [
        'find', 'expose', 'discover', 'unfamiliar', 'miss', 'hard', 'difficult', 'search'
    ],
    'Informal Distribution Barriers': [
        'consistency', 'platforms', 'advertise', 'events', 'promotion', 'creative', 'push', 'time'
    ]
}

# Function to assign a bucket based on keywords
def assign_bucket(text):
    text_lower = text.lower()
    for bucket, keywords in buckets.items():
        if any(re.search(r'\b' + kw + r'\b', text_lower) for kw in keywords):
            return bucket
    return 'Uncategorized'

df['Bucket'] = df['qualitative_text'].apply(assign_bucket)

print("=== Qualitative Text Clustering ===\n")

for bucket in buckets.keys():
    print(f"[{bucket}]")
    bucket_quotes = df[df['Bucket'] == bucket]['qualitative_text'].tolist()
    if bucket_quotes:
        for i, quote in enumerate(bucket_quotes[:3], 1):
            # Clean up the output string slightly for readability
            clean_quote = quote.replace(" | No response", "").replace("No response | ", "").strip(" |")
            print(f"  {i}. \"{clean_quote}\"")
    else:
        print("  (No quotes found for this bucket)")
    print("\n")
