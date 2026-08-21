"""
Module Name and Role: nlp_analysis.py - Component of notebooks/nlp_analysis.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json

df = pd.read_csv('../data/processed/fan_survey_cleaned.csv')

results = {}

# 1. Sentiment Analysis on `rec_system_feedback`
# Filter out "No response" and NAs
df_sent = df.dropna(subset=['rec_system_feedback', 'rec_system_rating']).copy()
df_sent = df_sent[~df_sent['rec_system_feedback'].str.contains('No response', case=False, na=False)]

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df_sent['feedback_sentiment'] = df_sent['rec_system_feedback'].apply(get_sentiment)

# Correlate sentiment with numeric rating (Pearson and Spearman)
correlation_pearson = df_sent['feedback_sentiment'].corr(df_sent['rec_system_rating'], method='pearson')
correlation_spearman = df_sent['feedback_sentiment'].corr(df_sent['rec_system_rating'], method='spearman')

results['sentiment_analysis'] = {
    'n_analyzed': len(df_sent),
    'mean_sentiment': float(df_sent['feedback_sentiment'].mean()),
    'correlation_with_rating_pearson': float(correlation_pearson),
    'correlation_with_rating_spearman': float(correlation_spearman)
}

# 2. Topic Modeling (LDA) on `challenge_following_local_artists`
df_chal = df.dropna(subset=['challenge_following_local_artists']).copy()
df_chal = df_chal[~df_chal['challenge_following_local_artists'].str.contains('No response', case=False, na=False)]

texts = df_chal['challenge_following_local_artists'].tolist()

# Vectorize
vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=0.9)
X = vectorizer.fit_transform(texts)
words = vectorizer.get_feature_names_out()

# Fit LDA
n_topics = 3
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
lda.fit(X)

topics_output = {}
for i, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[:-6:-1] # Top 5 words
    top_words = [words[idx] for idx in top_indices]
    topics_output[f"Topic_{i+1}"] = top_words

results['topic_modeling_challenges'] = {
    'n_analyzed': len(texts),
    'n_topics': n_topics,
    'top_words_per_topic': topics_output
}

# Save results
with open('nlp_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("NLP Analysis complete. Wrote to nlp_results.json")
