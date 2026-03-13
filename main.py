from preprocessing.preprocessor import (
    FULLY_PROCESSED_PARQUET, load_and_heal_data, homogenize_playlists,
    filter_entities, expand_features, remove_stop_words
)
from clustering.clustering_orchestrator import run_kmeans_pipeline, run_birch_pipeline
from evaluation.evaluator import eval

def main():
    # Preprocessing
    df = load_and_heal_data()
    df = homogenize_playlists(df)
    df = filter_entities(df)
    df = remove_stop_words(df)
    df = expand_features(df)

    # Clustering Orchestration
    # CHOOSE WHICH ALGO
    df, cluster_col = run_birch_pipeline(df, preset_k=55, verbose=True)

    # Persistence
    if cluster_col and cluster_col in df.columns:
        df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)

    # Evaluation
    print(f"\nStarting Evaluation on {cluster_col}...")
    eval(df=df, cluster_col=cluster_col, sample_frac=0.01)

if __name__ == "__main__":
    main()