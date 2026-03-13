import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from clustering.birch.birch_clustering import apply_birch
from clustering.kmeans.WCSS.WCSS import calculate_and_graph_wcss
from clustering.kmeans.kmeans_clustering import apply_k_means

def run_kmeans_pipeline(df, preset_k=None, verbose=True):
    """
    Orchestrates the entire KMeans workflow: TF-IDF -> WCSS (optional) -> Clustering.
    """
    def vprint(msg):
        if verbose: print(f"[DEBUG] {msg}")

    vprint("Extracting unique contextual features...")
    # Isolate unique features for the matrix
    contextual_mask = df['is_contextual'] == True
    unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()

    if len(unique_texts) == 0:
        print("[ERROR] No unique texts found for clustering.")
        return df, None

    print(f"Creating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(unique_texts)
    vprint(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    # Determine K
    if preset_k is not None:
        print(f"Using PRESET_K: {preset_k}")
        optimal_k = preset_k
    else:
        print("Running WCSS to find optimal K...")
        optimal_k = calculate_and_graph_wcss(tfidf_matrix=tfidf_matrix)
        if not optimal_k:
            print("[WARNING] WCSS failed. Defaulting to K=55.")
            optimal_k = 55

    target_col = f'k-means_cluster_{optimal_k}'

    # Check for existing results
    if target_col in df.columns:
        print(f"Cluster tags '{target_col}' already exist. Skipping.")
    else:
        vprint(f"Applying K-Means with k={optimal_k}...")
        df, _ = apply_k_means(df, unique_texts, tfidf_matrix, k=optimal_k)
    
    return df, target_col

def run_birch_pipeline(df, preset_k=None, verbose=True):
    """
    Orchestrates the entire BIRCH workflow: TF-IDF -> Clustering.
    """
    def vprint(msg):
        if verbose: print(f"[DEBUG] {msg}")

    vprint("Extracting unique contextual features...")
    # Isolate unique features for the matrix
    contextual_mask = df['is_contextual'] == True
    unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()

    if len(unique_texts) == 0:
        print("[ERROR] No unique texts found for clustering.")
        return df, None

    print(f"Creating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(unique_texts)
    vprint(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    # Determine K
    if preset_k is not None:
        print(f"Using PRESET_K: {preset_k}")
        optimal_k = preset_k
    else:
        print("No preset K provided. Exiting.")
        return Exception("BIRCH pipeline requires a preset K value. Please provide one.")

    target_col = f'birch_cluster_{optimal_k}'

    # Check for existing results
    if target_col in df.columns:
        print(f"Cluster tags '{target_col}' already exist. Skipping.")
    else:
        vprint(f"Applying BIRCH with k={optimal_k}...")
        df = apply_birch(df, unique_texts, tfidf_matrix, k=optimal_k)
    
    return df, target_col