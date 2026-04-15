import glob
import os

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
    # Shared tf-idf setup for all algorithms
    ##########################
    print("\nExtracting unique contextual features for TF-IDF matrix...")
    contextual_mask = df['is_contextual'] == True
    unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()
    print(f"Creating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, max_features=5678)
    tfidf_matrix = vectorizer.fit_transform(unique_texts)

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
    # Build the shared k-NN graph once

    # IMPORTANT REMEMBER THIS FOR THE NOTEBOOK, MOVE AFTER CLUSTERING 

    ##########################
    graph_builder = KNNGraph(k_neighbors=10, sim_threshold=0.15)
    print("\n[INFO] Building shared k-NN graph for graph-based clustering...")
    graph_builder.build_graph(tfidf_matrix, unique_texts)

    ##########################
    # Clustering & Graph Orchestration
    ##########################
    graph_config_name = f"k{graph_builder.k}_sim{graph_builder.sim_threshold}"
    algos = [
        KMeansClustering(k=55, max_iter=300, n_init=10, random_state=42),
        BirchClustering(k=55, threshold=0.9, branching_factor=25, batch_size=1000),
        BiclusteringAlgorithm(n_row_clusters=55, n_column_clusters=10, random_state=42),
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