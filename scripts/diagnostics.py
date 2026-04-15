import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Import your exact preprocessing pipeline
from preprocessing.preprocessor import (
    load_and_heal_data, homogenize_playlists,
    filter_entities, expand_features, remove_stop_words
)

# 1. Run the Preprocessing
print("Loading and preprocessing data...")
df = load_and_heal_data()
df = homogenize_playlists(df)
df = filter_entities(df)
df = remove_stop_words(df)
df = expand_features(df)

# 2. Extract texts and build TF-IDF exactly as main.py does
print("\nExtracting unique contextual features for TF-IDF matrix...")
contextual_mask = df['is_contextual'] == True
unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()

print(f"Creating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, max_features=5678)
tfidf_matrix = vectorizer.fit_transform(unique_texts)

# ==========================================
# 3. RUN DIAGNOSTICS
# ==========================================
print("\n" + "="*50)
print("🔍 TF-IDF MATRIX DIAGNOSTICS")
print("="*50)

print(f"Matrix Shape (Playlists x Words): {tfidf_matrix.shape}")

# Calculate the sum of every row. 
# We flatten() it to easily count the zeros.
row_sums = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

# Find the exact indices where the sum is exactly 0
empty_row_indices = np.where(row_sums == 0)[0]
num_empty_rows = len(empty_row_indices)
percentage_empty = (num_empty_rows / tfidf_matrix.shape[0]) * 100

print(f"Total Unique Contexts:   {tfidf_matrix.shape[0]:,}")
print(f"Empty Rows (All Zeros):  {num_empty_rows:,}")
print(f"Percentage Empty:        {percentage_empty:.3f}%")

# If there are empty rows, let's look at what got deleted!
if num_empty_rows > 0:
    print("\n⚠️ EXAMPLES OF 'ZEROED OUT' CONTEXTS:")
    print("These are the original texts that were completely stripped because")
    print("their words didn't make it into the top 5,678 features:")
    print("-" * 50)
    
    # Print up to 10 examples of the text that caused the NaN error
    for idx in empty_row_indices[:10]:
        print(f" - {unique_texts[idx]}")

print("="*50)