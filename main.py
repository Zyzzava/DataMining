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
from clustering.proclus.proclus_clustering import ProclusClustering

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
    # Clustering Orchestration
    # CHOOSE WHICH ALGO
    ##########################
    algos = [
        KMeansClustering(k=55, max_iter=300, n_init=10, random_state=42),
        BirchClustering(k=55, threshold=0.9, branching_factor=25, batch_size=1000),
        #ProclusClustering(k=55, l=5, random_state=42),
    ]

    for algo in algos:
        print(f"\n{'='*50}")
        print(f"Executing Pipeline for: {algo.algo_name}")
        print(f"{'='*50}")

        # Run the math
        df, cluster_col = algo.run_pipeline(df, unique_texts, tfidf_matrix)

        # Generate isolated reports & graphs
        algo.create_report()

        # Persistence (Save progress after each algorithm finishes)
        if cluster_col and cluster_col in df.columns:
            print(f"[INFO] Saving updated dataset to {FULLY_PROCESSED_PARQUET}...")
            df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)

    ##########################
    # Evaluation
    ##########################
    if cluster_col:
        print(f"\n[INFO] Starting Evaluation on {cluster_col}...")
        eval(df=df, cluster_col=cluster_col, unique_texts=unique_texts, tfidf_matrix=tfidf_matrix, sample_frac=0.01, output_dir=algo.report_dir)

if __name__ == "__main__":
    main()