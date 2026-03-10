import os
import re
import pandas as pd
import numpy as np
import spacy
from homogenitization import homogenize_series
from entity_filtering import is_contextual_playlist, setup_knowledge_base
from feature_expansion import expand_feature
from WCSS import *
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_FILE = 'spotify_dataset.csv'
FIXED_CSV = 'final_fixed.csv'
FINAL_PARQUET = 'spotify_final_healed.parquet'
FULLY_PROCESSED_PARQUET = 'spotify_fully_processed.parquet'
RANDOM_SEED = 42

def fix_nested_quotes(line):
    # Fixes internal quotes without breaking CSV structure
    line = line.replace('"""', '"')
    return re.sub(r'(?<!^)(?<!,)"(?!,)(?!$)', '""', line)

def create_gold_file():
    """Cleans, heals, and saves the dataset to Parquet format."""
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
    
    # Cleanup intermediate file to save space
    if os.path.exists(FIXED_CSV):
        os.remove(FIXED_CSV)
        
    return data

def load_data():
    """The only function you'll actually call in your other files."""
    if not os.path.exists(FINAL_PARQUET):
        return create_gold_file()
    return pd.read_parquet(FINAL_PARQUET)


if __name__ == "__main__":
    # 1. Load the Data
    df = load_data()
    print(f"Ready! Total rows: {len(df):,}")

    # 2. Standardization & Homogenization
    if 'homogenized_playlist' in df.columns:
        print("Column 'homogenized_playlist' already exists. Skipping homogenization.")
    else:
        df.columns = [c.strip().replace('"', '') for c in df.columns]
        print(f"Cleaned Columns: {df.columns.tolist()}")
        df['homogenized_playlist'] = homogenize_series(df['playlistname'])
        df.to_parquet(FINAL_PARQUET, index=False)
        print("Homogenization complete and saved.")

    # 3. Entity Filtering with spaCy
    if os.path.exists(FULLY_PROCESSED_PARQUET):
        print("\n[INFO] 'spotify_fully_processed.parquet' already exists. Loading it directly...")
        df = pd.read_parquet(FULLY_PROCESSED_PARQUET)
    else:
        # knowledge base for filtering
        known_artists, known_tracks, known_genres = setup_knowledge_base(FINAL_PARQUET)

        # Get unique playlist names for filtering
        unique_playlists = df['homogenized_playlist'].unique()
        
        print("\nLoading spaCy Language Model (en_core_web_lg)...")
        nlp = spacy.load("en_core_web_lg", disable=["parser", "attribute_ruler", "lemmatizer"])
        # Create a mapping of playlist name to whether it's contextual or not
        results_map = {
            name: is_contextual_playlist(str(name), nlp, known_artists=known_artists, known_tracks=known_tracks, known_genres=known_genres) 
            for name in tqdm(unique_playlists, desc="Entity Filtering")
        }

        # Map results back to the full dataset
        print("\n[3/3] Mapping results back to master dataset...")
        df['is_contextual'] = df['homogenized_playlist'].map(results_map)
        print("\n--- Results ---")
        # Print 10 unique random, where we see the playlist name and whether it's contextual or not
        print(df[df['is_contextual'] == True][['playlistname', 'homogenized_playlist']].head(10))
        
        # Final Save
        df.to_parquet('spotify_fully_processed.parquet', index=False)
    
    # Final check
    print(f"\nFinal dataset ready with {len(df):,} rows and {len(df.columns)} columns.")

    # How many true vs false?
    print("\nContextual vs Non-Contextual Counts:")
    print(df['is_contextual'].value_counts())

    # Printing for 10 unique rows with non-contextual playlists
    print("\nSample of Non-Contextual Playlists:")
    print(df[df['is_contextual'] == False].drop_duplicates(subset=['homogenized_playlist']).head(10))

    # 4. Feature Expansion for Contextual Playlists
    if 'expanded_features' not in df.columns:
        print("\nExpanding features for contextual playlists...")
        # Get unique contextual playlist names
        unique_contextual_playlists = df[df['is_contextual'] == True]['homogenized_playlist'].unique()
        # Expand features for unique names
        expanded_features_map = {
            name: expand_feature(name) for name in tqdm(unique_contextual_playlists, desc="Expanding Features")
        }
        # Map the expanded features back to the full dataset
        df['expanded_features'] = df['homogenized_playlist'].map(expanded_features_map)
        df.to_parquet('spotify_fully_processed.parquet', index=False)
    else: 
        print("\nExpanded features already exist. Skipping expansion step.")
        # Print 10 unqiue random, where we see the playlist name and the expanded features
        print("\nSample of Expanded Features:")
        sample_expanded = df[df['is_contextual'] == True][['homogenized_playlist', 'expanded_features']].drop_duplicates(subset=['homogenized_playlist']).sample(10, random_state=RANDOM_SEED)
        print(sample_expanded)

    # 5. Clustering Analysis with WCSS
    print("\nCalculating WCSS to determine optimal number of clusters...")
    # use df apply to calculate WCSS and graph it
    optimal_k, wcss, wcss_delta, avg_delta, std_delta, k_range = calculate_wcss(df['expanded_features'], sample_frac=1.00)
    # Check if WCSS folder exists, if not create it
    if not os.path.exists('WCSS'):
        os.makedirs('WCSS')
    graph_wcss(wcss, k_range, title_suffix=f"(Optimal k={optimal_k})")
    # but have shapes (98,) and (97,)
    threshold = avg_delta-std_delta
    graph_delta_wcss(wcss_delta, k_range, avg_delta, std_delta, threshold, title_suffix=f"(Delta WCSS)")
