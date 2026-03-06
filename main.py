import os
import re
import pandas as pd
import numpy as np
from homogenitization import homogenize_series
from entity_filtering import is_contextual_playlist
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_FILE = 'spotify_dataset.csv'
FIXED_CSV = 'final_fixed.csv'
FINAL_PARQUET = 'spotify_final_healed.parquet'
FULLY_PROCESSED_PARQUET = 'spotify_fully_processed.parquet'

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

    # WE ONLY DO THIS IF 'spotify_fully_processed.parquet' DOES NOT EXIST, OTHERWISE WE ASSUME IT'S BEEN DONE AND JUST LOAD IT
    if os.path.exists(FULLY_PROCESSED_PARQUET):
        print("\n[INFO] 'spotify_fully_processed.parquet' already exists. Loading it directly...")
        df = pd.read_parquet(FULLY_PROCESSED_PARQUET)
    else:
        # 3. SET UP KNOWLEDGE BASE (Loading Bar 1)
        print("\n[1/3] Building Knowledge Base for Filter...")
        # We use only unique values to keep the Set size manageable and fast
        known_artists = set(tqdm(df['artistname'].str.lower().dropna().unique(), desc="Indexing Artists"))
        
        known_genres = {
            'rock', 'pop', 'hip hop', 'rap', 'jazz', 'country', 'classical', 
            'metal', 'edm', 'r&b', 'indie', 'dance', 'house', 'techno'
        }

        # 4. UNIQUE ENTITY FILTERING (Loading Bar 2)
        # Optimization: Don't run spaCy 13 million times! Run it once per unique name.
        print(f"\n[2/3] Identifying unique playlists for Entity Filtering...")
        unique_playlists = df['homogenized_playlist'].unique()
        
        print(f"Analyzing {len(unique_playlists):,} unique names with spaCy (en_core_web_lg)...")
        
        # Process unique names with a progress bar
        # We pass the sets into the function as required
        results_map = {
            name: is_contextual_playlist(str(name), known_artists, known_genres, known_genres=known_genres) 
            for name in tqdm(unique_playlists, desc="Entity Filtering")
        }

        # 5. MAPPING RESULTS (Loading Bar 3)
        print("\n[3/3] Mapping results back to master dataset...")
        df['is_contextual'] = df['homogenized_playlist'].map(results_map)

        # 6. Final Results
        print("\n--- Results ---")
        # Show only the kept ones for the preview
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
    