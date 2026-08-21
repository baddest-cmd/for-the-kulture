"""
Module Name and Role: clean_data.py - Component of src/clean_data.py
Core Mathematical Inputs & Outputs: Inputs: N-dimensional data structures. Outputs: Transformed feature sets or metrics.
System Constraints: Must maintain deterministic execution and avoid state mutation where possible.
"""

import pandas as pd
import re
from pathlib import Path


# Define project root dynamically. This script is in 'src/', so the root is its parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define robust paths relative to the project root
raw_data_path = PROJECT_ROOT / 'data' / 'raw' / 'fan_survey.csv'
cleaned_data_path = PROJECT_ROOT / 'data' / 'raw' / 'fan_survey_cleaned.csv'

df = pd.read_csv(raw_data_path)

# Drop rows where all columns are missing
df.dropna(how='all', inplace=True)

# Rename columns to snake_case and shorter names
col_mapping = {
    'Timestamp': 'timestamp',
    'Are you based in South Africa?': 'based_in_sa',
    'Do you use a music streaming platform (e.g. YouTube Music, Spotify, Apple Music, Deezer) at least a few times a week?': 'uses_streaming',
    'Which music streaming platform(s) do you use most? (Select all that apply)': 'primary_platforms',
    'Roughly how many hours a week do you spend listening to music via streaming?': 'hours_streaming_weekly',
    'What genres do you listen to most? (Select all that apply)': 'top_genres',
    'In the past month, have you discovered a new artist you hadn\'t heard before?': 'discovered_new_artist_recently',
    'The last time I discovered a new artist, it was through:': 'discovery_method',
    'The last time I found a new artist I loved, it happened like this...': 'discovery_story',
    'Thinking about local genres like Amapiano or SA house specifically, I mostly discover new artists in this genre through:': 'local_discovery_method',
    'How well do you feel your streaming app\'s recommendations understand your taste in local/South African music?': 'rec_system_rating',
    'Please explain your rating above.': 'rec_system_feedback',
    'Have you used any AI-powered music features (e.g. AI radio, conversational music assistants, AI playlists)?': 'used_ai_features',
    'If you answered yes to the previous question, describe your experience using that AI feature.': 'ai_feature_experience',
    'If an AI assistant could help you with your music fandom in any way, what would you want it to do?': 'desired_ai_features',
    'The hardest part about following new/independent South African artists is...': 'challenge_following_local_artists',
    'Age band': 'age_band',
    'Province': 'province'
}

df.rename(columns=col_mapping, inplace=True)

# Convert timestamp to datetime (handling different formats if present)
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

# Fill missing values for text columns where appropriate
text_cols = ['discovery_story', 'rec_system_feedback', 'ai_feature_experience', 'desired_ai_features', 'challenge_following_local_artists']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("No response")

# Clean text columns: remove excessive newlines/carriage returns and trailing whitespace
def clean_text(x):
    if isinstance(x, str):
        # Replace newlines with spaces and strip
        return re.sub(r'[\r\n]+', ' ', x).strip()
    return x

for col in df.select_dtypes(include=['object']):
    df[col] = df[col].apply(clean_text)

# Fill other categorical missing values with 'Unknown'
cat_cols = ['based_in_sa', 'uses_streaming', 'primary_platforms', 'hours_streaming_weekly', 'top_genres', 
            'discovered_new_artist_recently', 'discovery_method', 'local_discovery_method', 
            'used_ai_features', 'age_band', 'province']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")
        
# For rec_system_rating (numeric or mixed), we can leave it as is or fill with median/mode, but we will fill with -1 or string 'Unknown' for now
if 'rec_system_rating' in df.columns:
    df['rec_system_rating'] = pd.to_numeric(df['rec_system_rating'], errors='coerce')

print("Cleaned shape:", df.shape)
print("Columns:", list(df.columns))
print("\nMissing values after clean:")
print(df.isnull().sum())

# Save to a new CSV
df.to_csv(cleaned_data_path, index=False)
print(f"\nSaved to '{cleaned_data_path}'")
