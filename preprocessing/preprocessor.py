import os
import re
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm

# Assuming these are in the same 'preprocessing' folder
from preprocessing.homogenitization import *
from preprocessing.entity_filtering import *
from preprocessing.feature_expansion import *
from preprocessing.stop_word_filtering import *

# --- CONFIGURATION ---
# Everything is in the data folder
RAW_FILE = 'data/spotify_dataset.csv'
FIXED_CSV = 'data/final_fixed.csv'
FINAL_PARQUET = 'data/spotify_final_healed.parquet'
FULLY_PROCESSED_PARQUET = 'data/spotify_fully_processed.parquet'

def fix_nested_quotes(line):
    """Fixes internal quotes without breaking CSV structure."""
    line = line.replace('"""', '"')
    return re.sub(r'(?<!^)(?<!,)"(?!,)(?!$)', '""', line)

def load_and_heal_data():
    """Cleans, heals, and loads the dataset into a DataFrame."""
    if os.path.exists(FINAL_PARQUET):
        return pd.read_parquet(FINAL_PARQUET)

    print(f"Phase 1: Regex cleaning {RAW_FILE}...")
    with open(RAW_FILE, 'r', encoding='utf-8') as f_in, \
         open(FIXED_CSV, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            f_out.write(fix_nested_quotes(line))

    healed_rows = []
    
    def bad_line_handler(line_cols):
        if len(line_cols) == 5:
            line_cols[2] = f"{line_cols[2]},{line_cols[3]}"
            del line_cols[3]
            healed_rows.append(line_cols)
        return None

    print(f"Phase 2: Loading and healing CSV...")
    data = pd.read_csv(
        FIXED_CSV, 
        doublequote=True, 
        on_bad_lines=bad_line_handler, 
        engine='python'
    )

    if healed_rows:
        print(f"Phase 3: Merging {len(healed_rows)} recovered rows...")
        healed_df = pd.DataFrame(healed_rows, columns=data.columns)
        data = pd.concat([data, healed_df], ignore_index=True)

    print(f"Phase 4: Exporting to {FINAL_PARQUET}...")
    data.to_parquet(FINAL_PARQUET, index=False)
    
    if os.path.exists(FIXED_CSV):
        os.remove(FIXED_CSV)
        
    return data

def homogenize_playlists(df):
    """Standardizes playlist names."""
    if 'homogenized_playlist' in df.columns:
        print("Column 'homogenized_playlist' already exists. Skipping homogenization.")
        return df

    df.columns = [c.strip().replace('"', '') for c in df.columns]
    print(f"Cleaned Columns: {df.columns.tolist()}")
    df['homogenized_playlist'] = homogenize_series(df['playlistname'])
    df.to_parquet(FINAL_PARQUET, index=False)
    print("Homogenization complete and saved.")
    return df

def filter_entities(df):
    """Filters contexts using spaCy and a knowledge base."""
    if os.path.exists(FULLY_PROCESSED_PARQUET):
        print("\n[INFO] 'spotify_fully_processed.parquet' already exists. Loading it directly...")
        return pd.read_parquet(FULLY_PROCESSED_PARQUET)

    known_artists, known_genres = setup_knowledge_base(FINAL_PARQUET)
    unique_playlists = df['homogenized_playlist'].unique()
    
    print("\nLoading spaCy Language Model (en_core_web_lg)...")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "attribute_ruler", "lemmatizer"])
    
    results_map = {
        name: is_contextual_playlist(str(name), nlp, known_artists=known_artists, known_genres=known_genres) 
        for name in tqdm(unique_playlists, desc="Entity Filtering")
    }

    print("\nMapping results back to master dataset...")
    df['is_contextual'] = df['homogenized_playlist'].map(results_map)
    df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)
    return df

def expand_features(df):
    """Expands features for contextual playlists."""
    if 'expanded_features' in df.columns:
        print("\nExpanded features already exist. Skipping expansion step.")
        return df

    print("\nExpanding features for contextual playlists...")
    
    unique_contextual_playlists = df[df['is_contextual'] == True]['filtered_playlist'].unique()
    
    expanded_features_map = {
        name: expand_feature(name) for name in tqdm(unique_contextual_playlists, desc="Expanding Features")
    }
    
    df['expanded_features'] = df['filtered_playlist'].map(expanded_features_map)
    df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)
    
    return df

def remove_stop_words(df):
    """Removes stop words from homogenized playlists."""
    df = filter_stop_words_step(df, FULLY_PROCESSED_PARQUET)
    return df