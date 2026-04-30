import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

def evaluate_silhouette(df, unique_texts, tfidf_matrix, cluster_col='k-means_cluster', sample_size=5000):
    print(f"\n{'='*30}\nCALCULATING SILHOUETTE SCORE FOR {cluster_col}\n{'='*30}")

    # Use only contextual rows with real labels so non-contextual NaNs do not overwrite valid assignments.
    contextual_df = df[df['is_contextual'] == True]
    labeled_contexts = contextual_df.dropna(subset=['expanded_features', cluster_col])
    labeled_contexts = labeled_contexts.drop_duplicates(subset=['expanded_features'])

    text_to_cluster = dict(zip(labeled_contexts['expanded_features'], labeled_contexts[cluster_col]))

    valid_mask = np.array([text in text_to_cluster and pd.notna(text_to_cluster[text]) for text in unique_texts])
    
    if not np.any(valid_mask): # changed to np.any() since it's an array now
        print(f"-> Skipping silhouette score for {cluster_col}: no valid labels found.")
        return

    # Now SciPy will accept this filter safely!
    filtered_matrix = tfidf_matrix[valid_mask]
    ordered_labels = [text_to_cluster[text] for text in unique_texts if text in text_to_cluster and pd.notna(text_to_cluster[text])]

    if len(set(ordered_labels)) < 2:
        print(f"-> Skipping silhouette score for {cluster_col}: need at least 2 clusters, found {len(set(ordered_labels))}.")
        return

    # silhouette score with cosine because tf-idf vectors are sparse and high-dimensional
    score = silhouette_score(filtered_matrix, ordered_labels, metric='cosine', sample_size=sample_size, random_state=42)

    print(f"-> Final Silhouette Score ({cluster_col}): {score:.4f}")