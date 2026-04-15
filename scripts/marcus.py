import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_average_compactness(texts, matrix, text_to_cluster_dict):
    # Extract labels in the exact same order as the TF-IDF matrix rows
    labels = np.array([text_to_cluster_dict.get(t, -1) for t in texts])
    
    cluster_sims = []
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1 or pd.isna(cluster_id): # Skip noise / unassigned
            continue
            
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
            
        # Get points for the cluster
        cluster_points = matrix[idx]
        
        # Calculate the centroid (average TF-IDF vector of the cluster)
        centroid = cluster_points.mean(axis=0)
        
        # Calculate cosine similarity of all points to the centroid
        sims = cosine_similarity(cluster_points, np.asarray(centroid))
        
        # Save the average similarity for this cluster
        cluster_sims.append(sims.mean())
        
    # Return the overall average compactness across all valid clusters
    return np.mean(cluster_sims)

print("Calculating intra-cluster compactness...")

# 1. Create dictionaries mapping the unique texts back to their assigned clusters
text_to_kmeans = dict(zip(df['expanded_features'], df[k_means_55_col]))
text_to_birch = dict(zip(df['expanded_features'], df[birch_55_col]))

# 2. Calculate and print
kmeans_compactness = get_average_compactness(unique_texts, tfidf_matrix, text_to_kmeans)
birch_compactness = get_average_compactness(unique_texts, tfidf_matrix, text_to_birch)

print(f"K-Means Average Intra-Cluster Cosine Similarity: {kmeans_compactness:.4f}")
print(f"BIRCH Average Intra-Cluster Cosine Similarity:   {birch_compactness:.4f}")