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