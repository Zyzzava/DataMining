#!/usr/bin/env python3
"""
Usage:
  python scripts/evaluate_cluster.py data/spotify_fully_processed.parquet louvain_community_labels
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import TruncatedSVD

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_cluster.py <data_path> <cluster_col>")
        sys.exit(1)

    data_path = sys.argv[1]
    cluster_col = sys.argv[2]

    # 1. Load data
    print(f"Loading Data: {data_path}")
    df = pd.read_parquet(data_path)

    # 2. CRITICAL FIX: Filter for contextual rows AND valid cluster labels
    # This prevents the "Input y contains NaN" error
    print(f"Filtering for rows with valid labels in '{cluster_col}'...")
    df_labeled = df[df[cluster_col].notna()].copy()
    
    if df_labeled.empty:
        print(f"Error: No valid labels found in column '{cluster_col}'.")
        sys.exit(1)

    # Group by the expanded features to match your clustering logic
    # Clustering was done on unique texts, so we evaluate on unique texts
    unique_df = df_labeled.drop_duplicates('expanded_features')
    unique_texts = unique_df['expanded_features'].values
    labels = unique_df[cluster_col].values

    # 3. Re-generate TF-IDF matrix (matching main.py config)
    print(f"Re-generating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, max_features=5678)
    tfidf_matrix = vectorizer.fit_transform(unique_texts)
    
    # 4. Silhouette Calculation with Sampling
    # replace 999999999999, with a reasonable sample size to avoid memory issues
    sample_size = min(999999999999, tfidf_matrix.shape[0])
    print(f"Calculating Silhouette (Sampling {sample_size:,} for speed)...")
    
    indices = np.random.choice(tfidf_matrix.shape[0], sample_size, replace=False)
    X_sample = tfidf_matrix[indices]
    labels_sample = labels[indices]
    
    # Calculate score
    avg_score = silhouette_score(X_sample, labels_sample, metric='cosine')
    print(f"Average Silhouette Score ({cluster_col}): {avg_score:.4f}")

    # 5. Visualizations
    print("Generating Silhouette Plot...")
    sample_values = silhouette_samples(X_sample, labels_sample, metric='cosine')
    plt.figure(figsize=(12, 8))
    y_lower = 10
    sorted_clusters = sorted(np.unique(labels_sample))
    
    for i in sorted_clusters:
        ith_values = sample_values[labels_sample == i]
        ith_values.sort()
        y_upper = y_lower + len(ith_values)
        color = plt.cm.nipy_spectral(float(i) / len(sorted_clusters))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values, 
                          facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 10

    plt.axvline(x=avg_score, color="red", linestyle="--")
    plt.title(f"Silhouette Analysis for {cluster_col}\nAvg Score: {avg_score:.4f}")
    plt.ylabel("Cluster Label")
    plt.xlabel("Silhouette Coefficient")
    plt.savefig(f"silhouette_{cluster_col}.png")

    print("Generating 2D Projection...")
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X_sample)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_sample, cmap='tab20', s=2, alpha=0.6)
    plt.title(f"2D Projection: {cluster_col}")
    plt.colorbar(label='Cluster ID')
    plt.savefig(f"viz_{cluster_col}.png")
    
    print(f"Done. Saved: silhouette_{cluster_col}.png and viz_{cluster_col}.png")

if __name__ == '__main__':
    main()