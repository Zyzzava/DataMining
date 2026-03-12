import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def apply_k_means(df, unique_texts, tfidf_matrix, text_column='expanded_features', k=30):
    """Applies KMeans clustering and maps labels to a dynamic column name."""
    print(f"Applying KMeans clustering... on {len(unique_texts)} unique contexts with k={k}")

    # Fit the final model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    # unique texts and cluster_labels should have the same length and order 
    cluster_mapping = dict(zip(unique_texts, cluster_labels))

    # Create the dynamic column name
    new_col_name = f'k-means_cluster_{k}'
    
    # Broadcast the clusters back to the DataFrame
    df[new_col_name] = df[text_column].map(cluster_mapping)
    print(f"Added '{new_col_name}' labels to the DataFrame.")
    
    return df, new_col_name

def run_sanity_check(df, k=30):
    """Prints a high-variety cluster sanity check."""
    print(f"\n{'='*30} HIGH-VARIETY CLUSTER SAMPLES {'='*30}")
    sampled_ids = np.random.choice(df[f'k-means_cluster_{k}'].dropna().unique(), 3, replace=False)

    for c_id in sorted(sampled_ids):
        cluster_df = df[df[f'k-means_cluster_{k}'] == c_id]
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