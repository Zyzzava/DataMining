from sklearn.metrics import silhouette_score

def evaluate_silhouette(df, unique_texts, tfidf_matrix, cluster_col='k-means_cluster', sample_size=5000):
    print(f"\n{'='*30}\nCALCULATING SILHOUETTE SCORE FOR {cluster_col}\n{'='*30}")

    #align labels
    text_to_cluster = dict(zip(df['expanded_features'], df[cluster_col]))

    #ordered list of labels
    ordered_labels = [text_to_cluster[text] for text in unique_texts]

    #silhouette score with cosine because tf-idf vectors are sparse and high-dimensional
    score = silhouette_score(tfidf_matrix, ordered_labels, metric='cosine', sample_size=sample_size, random_state=42)

    print(f"-> Final Silhouette Score ({cluster_col}): {score:.4f}")