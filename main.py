from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# data wrangling
from preprocessing.preprocessor import (
    FULLY_PROCESSED_PARQUET,
    load_and_heal_data,
    homogenize_playlists,
    filter_entities,
    expand_features,
    remove_stop_words
)
# modeling and evaluation tools
from clustering.kmeans.WCSS.WCSS import calculate_and_graph_wcss
from clustering.kmeans.kmeans_clustering import apply_k_means, run_sanity_check
from evaluation.evaluator import eval

# --- CONFIGURATION ---
VERBOSE = True  # Toggle this to False to hide all [DEBUG] prints

def vprint(msg):
    """Helper function to print debugging information if VERBOSE is True."""
    if VERBOSE:
        print(f"[DEBUG] {msg}")

def main():
    print("Starting Spotify Dataset Pre-Processing Pipeline...")
    
    # 1. Load Data
    vprint("STEP 1: Loading Data...")
    df = load_and_heal_data()
    print(f"Loaded. Total rows: {len(df):,}")
    vprint(f"DataFrame Shape: {df.shape}")
    vprint(f"Initial Columns: {df.columns.tolist()}")

    # 2. Standardization & Homogenization
    vprint("\nSTEP 2: Homogenizing Playlists...")
    df = homogenize_playlists(df)
    vprint(f"Columns after homogenization: {df.columns.tolist()}")
    if VERBOSE and 'homogenized_playlist' in df.columns:
        vprint(f"Nulls in 'homogenized_playlist': {df['homogenized_playlist'].isna().sum()}")
        vprint("Sample of homogenization:\n" + str(df[['playlistname', 'homogenized_playlist']].head(3)))

    # 3. Entity Filtering
    vprint("\nSTEP 3: Entity Filtering...")
    df = filter_entities(df)
    if VERBOSE and 'is_contextual' in df.columns:
        # Count unique contextual playlists
        unique_contextual = df[df['is_contextual'] == True]['homogenized_playlist'].nunique()
        # Count total unique playlists
        unique_total = df['homogenized_playlist'].nunique()
        
        vprint(f"Unique contextual playlists found: {unique_contextual:,} out of {unique_total:,} ({unique_contextual/unique_total:.1%})")

    # 4. Stop Word Removal
    vprint("\nSTEP 4: Removing Stop Words...")
    df = remove_stop_words(df)
    if VERBOSE and 'filtered_playlist' in df.columns:
        vprint(f"Nulls in 'filtered_playlist': {df['filtered_playlist'].isna().sum()}")
        # Check if the column is accidentally full of empty strings after filtering
        empty_strings = (df['filtered_playlist'].astype(str).str.strip() == "").sum()
        vprint(f"Empty strings in 'filtered_playlist': {empty_strings:,}")
        vprint("Sample of filtering (Contextual only):\n" + 
               str(df[df['is_contextual'] == True][['homogenized_playlist', 'filtered_playlist']].head(3)))

    # 5. Feature Expansion
    vprint("\nSTEP 5: Expanding Features...")
    df = expand_features(df)
    if VERBOSE and 'expanded_features' in df.columns:
        vprint(f"Nulls in 'expanded_features': {df['expanded_features'].isna().sum()}")
        vprint("Sample of expanded features (Contextual only):\n" + 
               str(df[df['is_contextual'] == True][['filtered_playlist', 'expanded_features']].head(3)))

    # SANITY CHECK #1
    print("\nSANITY CHECK #1: Current Columns")
    for col in df.columns:
        print(f" - {col}")

    # 6. K-Means Clustering & WCSS
    vprint("\nSTEP 6: K-Means Clustering Setup...")

    PRESET_K = 55  # Set to an integer to skip WCSS, or None to run it
    cluster_col_to_use = None

    # Step 6: K-Means Clustering Setup
    vprint("\nSTEP 6: K-Means Clustering Setup...")

    # Extract unique texts and build TF-IDF matrix (Needed for both paths)
    vprint("Extracting unique texts for TF-IDF...")
    unique_texts = df[df['is_contextual'] == True]['expanded_features'].dropna().unique()

    if len(unique_texts) > 0:
        print(f"\nCreating TF-IDF matrix for {len(unique_texts)} unique expanded features...")
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(unique_texts)
        vprint(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

        if PRESET_K is not None:
            print(f"\nPRESET_K is set to {PRESET_K}. Skipping WCSS.")
            optimal_k = PRESET_K
        else:
            print("\nNo PRESET_K provided. Running WCSS to determine optimal K.")
            # WCSS returns optimal_k, wcss, wcss_delta, avg_delta, std_delta, k_range
            # Based on your WCSS.py calculate_and_graph_wcss wrapper:
            optimal_k = calculate_and_graph_wcss(tfidf_matrix=tfidf_matrix)
            
            if not optimal_k:
                print("ERROR: Optimal K not determined. Using default 55.")
                optimal_k = 55

        # Check if this specific K-column already exists to avoid re-running
        target_col = f'k-means_cluster_{optimal_k}'
        if target_col not in df.columns:
            vprint(f"Applying K-Means with k={optimal_k}...")
            df, cluster_col_to_use = apply_k_means(df, unique_texts, tfidf_matrix, k=optimal_k)
            # Save progress
            df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)
        else:
            print(f"\nCluster tags '{target_col}' already exist. Skipping.")
            cluster_col_to_use = target_col

    # SANITY CHECK #2
    print("\nSANITY CHECK #2: Current Columns")
    for col in df.columns:
        print(f" - {col}")

    # Quality Assurance
    if 'k-means_cluster' in df.columns:
        # run_sanity_check(df)
        pass

    print("\n finished successfully")
    print("\nStarting Evaluation")

    # 7. Evaluation
    vprint("STEP 7: Calling eval() pipeline...")
    # Evaluate the model using the specified evaluation pipeline
    eval(df=df, cluster_col=target_col, sample_frac=0.01)

    # SANITY CHECK #3
    print("\nSANITY CHECK #3: Final Columns")
    for col in df.columns:
        print(f" - {col}")

if __name__ == "__main__":
    main()