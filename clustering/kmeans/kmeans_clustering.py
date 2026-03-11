import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def apply_k_means(df, unique_texts, tfidf_matrix, text_column='expanded_features', k=30):
    """Applies KMeans clustering to the TF-IDF matrix and maps the cluster labels back to the original DataFrame."""
    print(f"Applying KMeans clustering... on {len(unique_texts)} unique contexts")

    # Fit the final model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # unique texts and cluster_labels should have the same length and order 
    cluster_mapping = dict(zip(unique_texts, cluster_labels))

    # Broadcast the clusters back to a new Column in the original DataFrame
    df['k-means_cluster'] = df[text_column].map(cluster_mapping)
    print("Added 'k-means_cluster' labels to the DataFrame.")

    return df 

def apply_k_means_clusters(df, unique_texts, tfidf_matrix, k=30, output_file='spotify_fully_processed.parquet'):
    """Tags the dataset with final clusters and saves it."""
    if 'k-means_cluster' in df.columns:
        print("\nCluster tags already exist. Skipping final clustering.")
        return df

    print(f"\nAssigning contextual clusters (k={k}) to the dataset...")
    df = apply_k_means(df, unique_texts, tfidf_matrix, text_column='expanded_features', k=k)
    df.to_parquet(output_file, index=False)
    print(f"\nFinal dataset saved with 'k-means_cluster' tags to {output_file}!")
    return df

def run_sanity_check(df):
    """Prints a high-variety cluster sanity check."""
    print(f"\n{'='*30} HIGH-VARIETY CLUSTER SAMPLES {'='*30}")
    sampled_ids = np.random.choice(df['k-means_cluster'].dropna().unique(), 3, replace=False)

    for c_id in sorted(sampled_ids):
        cluster_df = df[df['k-means_cluster'] == c_id]
        all_unique_playlists = cluster_df['playlistname'].unique()
        print(f"\n[Cluster {int(c_id)}] ({len(all_unique_playlists):,} Unique Contexts)")
        
        unique_songs = cluster_df['trackname'].unique()
        np.random.shuffle(unique_songs)
        
        for song in unique_songs[:5]:
            associated_playlists = cluster_df[cluster_df['trackname'] == song]['playlistname'].unique()
            playlist_str = ", ".join(associated_playlists[:4])
            if len(associated_playlists) > 4:
                playlist_str += f" (+{len(associated_playlists)-4} more)"
            print(f"  🎵 {str(song)[:30]:<32} | Contexts: {playlist_str}")
    print(f"\n{'='*90}")