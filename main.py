import os
import re
import pandas as pd
import numpy as np
import spacy
from homogenitization import *
from entity_filtering import *
from feature_expansion import *
from WCSS import *
from clustering import *
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

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

    # 5. Clustering Analysis with WCSS if WCSS folder is empty

    # 5.1 Isolate UNIQUE expanded features safely
    # This prevents NameErrors and guarantees we only vectorize non-NaN expanded features
    unique_texts = df[df['is_contextual'] == True]['expanded_features'].dropna().unique()

    # 5.2 Create TF-IDF Matrix on the pre-aligned list
    print(f"\nCreating TF-IDF matrix for {len(unique_texts)} unique expanded features...")
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(unique_texts)

    # 5.3 WCSS Calculation
    if os.path.exists('WCSS') and os.listdir('WCSS'):
        print("\nWCSS graphs already exist. Skipping WCSS calculation and graphing.")
    else:
        print("\nCalculating WCSS to determine optimal number of clusters...")
        # Now we only pass the matrix!
        optimal_k, wcss, wcss_delta, avg_delta, std_delta, k_range = calculate_wcss(tfidf_matrix)
        
        if not os.path.exists('WCSS'):
            os.makedirs('WCSS')
        graph_wcss(wcss, k_range, title_suffix=f"(Optimal k={optimal_k})")
        
        threshold = avg_delta - std_delta
        graph_delta_wcss(wcss_delta, k_range, avg_delta, std_delta, threshold, title_suffix=f"(Delta WCSS)")

    # 6. Running final K-means (Tagging the dataset with k=30)
    if 'k-means_cluster' not in df.columns:
        print("\nAssigning contextual clusters (k=30) to the dataset...")
        
        # Pass BOTH the aligned text array and the matrix
        df = apply_k_means(df, unique_texts, tfidf_matrix, text_column='expanded_features', k=30)
        
        df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)
        print(f"\nFinal dataset saved with 'k-means_cluster' tags to {FULLY_PROCESSED_PARQUET}!")
    else:
        print("\nK-means cluster tags already exist. Skipping final clustering.")
        print("\nSample of Assigned Clusters:")
        print(df[df['is_contextual'] == True][['homogenized_playlist', 'k-means_cluster']].drop_duplicates().head(10))

    # --- High-Variety Cluster Sanity Check ---
    print(f"\n{'='*30} HIGH-VARIETY CLUSTER SAMPLES {'='*30}")
    # Pull 3 random clusters to inspect
    sampled_ids = np.random.choice(df['k-means_cluster'].dropna().unique(), 3, replace=False)

    for c_id in sorted(sampled_ids):
        # Filter for the specific cluster
        cluster_df = df[df['k-means_cluster'] == c_id]
        
        # Get unique playlists in this cluster to show total variety count
        all_unique_playlists = cluster_df['playlistname'].unique()
        
        print(f"\n[Cluster {int(c_id)}] ({len(all_unique_playlists):,} Unique Contexts)")
        
        # Get unique songs, but shuffle them to see different parts of the cluster
        unique_songs = cluster_df['trackname'].unique()
        np.random.shuffle(unique_songs)
        
        # Inspect 5 random songs and the unique playlist names they appear under
        for song in unique_songs[:5]:
            # Extract unique playlist names for this specific song
            associated_playlists = cluster_df[cluster_df['trackname'] == song]['playlistname'].unique()
            
            # Format the playlists for the printout (limit to 4 to keep it clean)
            playlist_str = ", ".join(associated_playlists[:4])
            if len(associated_playlists) > 4:
                playlist_str += f" (+{len(associated_playlists)-4} more)"
                
            print(f"  🎵 {str(song)[:30]:<32} | Contexts: {playlist_str}")

    print(f"\n{'='*90}")