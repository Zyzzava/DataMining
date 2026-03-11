from sklearn.feature_extraction.text import TfidfVectorizer

# data wrangling
from preprocessing.preprocessor import (
    load_and_heal_data,
    homogenize_playlists,
    filter_entities,
    expand_features,
)
# modeling and evaluation tools
from clustering.kmeans.WCSS.WCSS import calculate_and_graph_wcss
from clustering.kmeans.kmeans_clustering import apply_k_means, run_sanity_check
from evaluation.evaluator import eval

def main():
    print("Starting Spotify Dataset Pre-Processing Pipeline...")
    # Load the Data
    df = load_and_heal_data()
    print(f"Loaded. Total rows: {len(df):,}")

    # Standardization & Homogenization
    df = homogenize_playlists(df)

    # Entity Filtering with spaCy
    df = filter_entities(df)

    # Feature Expansion for Contextual Playlists
    df = expand_features(df)

    # Matrix Creation, WCSS, and Clustering
    if 'k-means_cluster' not in df.columns:
        # Isolate UNIQUE expanded features safely to avoid NaN errors
        unique_texts = df[df['is_contextual'] == True]['expanded_features'].dropna().unique()
        
        if len(unique_texts) > 0:
            print(f"\nCreating TF-IDF matrix for {len(unique_texts)} unique expanded features...")
            vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
            tfidf_matrix = vectorizer.fit_transform(unique_texts)

            # WCSS
            calculate_and_graph_wcss(tfidf_matrix)

            # Final KMeans Mapping
            df = apply_k_means(df, unique_texts, tfidf_matrix, k=30)
    else:
        print("\nCluster tags already exist. Skipping TF-IDF, WCSS, and clustering steps.")

    # Quality Assurance
    if 'k-means_cluster' in df.columns:
        run_sanity_check(df)

    print("\n finished successfully")
    print("\nStarting Evaluation")

    # Evaluate the model using the specified evaluation pipeline
    eval(cluster_col='k-means_cluster')

if __name__ == "__main__":
    main()