import glob
import os

# Saving data structs for future use.
import scipy.sparse
import pickle
#####################################

from clustering.kmeans.WCSS.WCSS import calculate_and_graph_wcss
from clustering.tf_idf_analysis.tf_idf_analysis import run_full_tfidf_analysis
from preprocessing.preprocessor import (
    FULLY_PROCESSED_PARQUET, load_and_heal_data, homogenize_playlists,
    filter_entities, expand_features, remove_stop_words
)
from evaluation.evaluator import eval
from sklearn.feature_extraction.text import TfidfVectorizer

# Clustering algorithm imports
from clustering.birch.birch_clustering import BirchClustering
from clustering.kmeans.kmeans_clustering import KMeansClustering
from clustering.biclustering.spectral_biclustering import BiclusteringAlgorithm
from clustering.coclustering.spectral_coclustering import CoclusteringAlgorithm
# from clustering.proclus.proclus_clustering import ProclusClustering # This don't work, do desktop

# Graph import 
from graph.knn.knn_graph import KNNGraph
from graph.knn.louvain_clustering import LouvainClustering
from graph.knn.spectral_clustering import SpectralGraphClustering

def main():
    ##########################
    # Preprocessing
    ##########################
    df = load_and_heal_data()
    df = homogenize_playlists(df)
    df = filter_entities(df)
    df = remove_stop_words(df)
    df = expand_features(df)

    ##########################
    # Shared tf-idf setup for all algorithms (With Caching & Filtering)
    ##########################
    tfidf_cache_dir = "data/tfidf_cache"
    os.makedirs(tfidf_cache_dir, exist_ok=True)
    
    # Define file paths for the cached objects
    matrix_path = os.path.join(tfidf_cache_dir, "cleaned_tfidf_matrix.npz")
    texts_path = os.path.join(tfidf_cache_dir, "cleaned_unique_texts.pkl")
    vectorizer_path = os.path.join(tfidf_cache_dir, "vectorizer.pkl")

    # Check if all cached files exist
    if os.path.exists(matrix_path) and os.path.exists(texts_path) and os.path.exists(vectorizer_path):
        print("\n[INFO] Loading cached, cleaned TF-IDF matrix and unique texts...")
        tfidf_matrix = scipy.sparse.load_npz(matrix_path)
        
        with open(texts_path, 'rb') as f:
            unique_texts = pickle.load(f)
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        print(f"Loaded TF-IDF matrix shape: {tfidf_matrix.shape}")

    else:
        print("\nExtracting unique contextual features for TF-IDF matrix...")
        contextual_mask = df['is_contextual'] == True
        unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()
        
        print(f"Creating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, max_features=5678)
        tfidf_matrix = vectorizer.fit_transform(unique_texts)

        # --- Noise Filtering Block ---
        import numpy as np
        row_sums = np.squeeze(np.asarray(tfidf_matrix.sum(axis=1)))
        non_empty_mask = row_sums > 0
        
        dropped_count = len(unique_texts) - non_empty_mask.sum()
        if dropped_count > 0:
            print(f"[INFO] Dropping {dropped_count:,} contexts that became empty after TF-IDF filtering (Noise).")
        
        # Apply the filter
        unique_texts = unique_texts[non_empty_mask]
        tfidf_matrix = tfidf_matrix[non_empty_mask]
        
        # --- Save to Cache ---
        print(f"[INFO] Saving cleaned TF-IDF matrix and objects to '{tfidf_cache_dir}'...")
        scipy.sparse.save_npz(matrix_path, tfidf_matrix)
        
        with open(texts_path, 'wb') as f:
            pickle.dump(unique_texts, f)
            
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    ##########################
    # If the TF-IDF hasn't been analysed, let's do that.
    ##########################
    if not os.path.exists('tf_idf_analysis') or not os.listdir('tf_idf_analysis'):
        print("\nRunning TF-IDF Analysis...")
        run_full_tfidf_analysis(tfidf_matrix, vectorizer)
    else:
        print("\nTF-IDF analysis already exists. Skipping TF-IDF analysis.")

    ##########################
    # WCSS / Optimal K Check
    ##########################
    wcss_dir = "clustering/kmeans/WCSS"
    os.makedirs(wcss_dir, exist_ok=True)
    
    # Search for any file ending in .png in the directory
    wcss_pngs = glob.glob(os.path.join(wcss_dir, "*.png"))
    
    optimal_k = 55 # A fallback default just in case
    optimal_k_file = os.path.join(wcss_dir, "optimal_k.txt")
    
    if not wcss_pngs:
        print(f"\nNo WCSS graphs found in '{wcss_dir}'. Running WCSS to find optimal K...")
        
        # Run the heavy calculation
        calculated_k = calculate_and_graph_wcss(tfidf_matrix=tfidf_matrix)
        
        if calculated_k:
            optimal_k = calculated_k
            # Save this number to a text file so we remember it for the next run!
            with open(optimal_k_file, "w") as f:
                f.write(str(optimal_k))
                
    else:
        print(f"\nWCSS graphs already exist in '{wcss_dir}'. Skipping heavy calculation.")
        
        # Try to load the previously saved optimal K
        if os.path.exists(optimal_k_file):
            with open(optimal_k_file, "r") as f:
                optimal_k = int(f.read().strip())
            print(f"Loaded previously calculated optimal K: {optimal_k}")
        else:
            print(f"No saved K value found. Defaulting to K={optimal_k}")

    ##########################
    # Build or Load the shared k-NN graph
    ##########################
    graph_builder = KNNGraph(k_neighbors=10, sim_threshold=0.15)
    
    # --- FIX: Add the dataset size (N) to the filename ---
    graph_config_name = f"k{graph_builder.k}_sim{graph_builder.sim_threshold}_N{len(unique_texts)}"
    
    graph_save_dir = os.path.join("graph", "knn", "saved_graphs")
    os.makedirs(graph_save_dir, exist_ok=True)
    
    graph_save_path = os.path.join(graph_save_dir, f"knn_{graph_config_name}.pkl")

    if os.path.exists(graph_save_path):
        print(f"\n[INFO] Loading previously built k-NN graph from {graph_save_path}...")
        with open(graph_save_path, "rb") as f:
            graph_builder.G = pickle.load(f)
    else:
        print("\n[INFO] Building shared k-NN graph for graph-based clustering...")
        graph_builder.build_graph(tfidf_matrix, unique_texts)
        
        print(f"[INFO] Saving generated k-NN graph to {graph_save_path}...")
        with open(graph_save_path, "wb") as f:
            pickle.dump(graph_builder.G, f)

    ##########################
    # Clustering & Graph Orchestration
    ##########################
    graph_config_name = f"k{graph_builder.k}_sim{graph_builder.sim_threshold}"
    algos = [
        KMeansClustering(k=55, max_iter=300, n_init=10, random_state=42),
        BirchClustering(k=55, threshold=0.9, branching_factor=25, batch_size=1000),
        BiclusteringAlgorithm(n_row_clusters=55, n_column_clusters=10, random_state=42),
        CoclusteringAlgorithm(n_clusters=55, random_state=42),
        LouvainClustering(graph=graph_builder.G, graph_config_name=graph_config_name),
        # normally optimal k used here, but 322 because louvain is finding 322 
        SpectralGraphClustering(graph=graph_builder.G, n_clusters=720, graph_config_name=graph_config_name),
    ]

    for algo in algos:
        target_col = getattr(algo, "cluster_col", None)
        report_out = algo.report_dir

        print(f"[DEBUG] Checking for: '{target_col}' in columns: {df.columns.tolist()[:10]}...")
        if target_col in df.columns:
            print(f"\n[SKIP] {algo.algo_name} already exists in column '{target_col}'.")
        else:
            print(f"\n{'='*50}")
            print(f"Executing Pipeline for: {algo.algo_name}")
            print(f"{'='*50}")

            df, target_col = algo.run_pipeline(df, unique_texts, tfidf_matrix)
            algo.create_report()

            if target_col and target_col in df.columns:
                print(f"[INFO] Saving updated labels to {FULLY_PROCESSED_PARQUET}...")
                df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)

        ##########################
        # Evaluation & Comparison Skip Logic
        ##########################
        if target_col in df.columns:
            os.makedirs(report_out, exist_ok=True)

            eval_report_path = os.path.join(report_out, f"evaluation_metrics_{target_col}.txt")

            if os.path.exists(eval_report_path):
                print(f"[SKIP] Evaluation for {target_col} already exists at: {eval_report_path}")
            else:
                print(f"\n[INFO] Starting Evaluation on {target_col}...")
                eval(
                    df=df,
                    cluster_col=target_col,
                    unique_texts=unique_texts,
                    tfidf_matrix=tfidf_matrix,
                    sample_frac=0.01,
                    output_dir=report_out,
                )

if __name__ == "__main__":
    main()