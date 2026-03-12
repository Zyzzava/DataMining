import numpy as np
from sklearn.cluster import Birch


def apply_birch(df, unique_texts, tfidf_matrix, k=34):
    print(f"Applying BIRCH clustering... on {len(unique_texts)} unique contexts")

    # Initialize BIRCH with the specified number of clusters
    birch = Birch(n_clusters=k)

    # Fit the model to the TF-IDF matrix
    cluster_labels = birch.fit_predict(tfidf_matrix)

    # zip
    cluster_mapping = list(zip(unique_texts, cluster_labels))

    #broadcast the cluster labels to the original dataframe
    df['birch_cluster'] = df['expanded_features'].map(dict(cluster_mapping))
    print("Added 'birch_cluster' labels to the DataFrame.")

    return df
