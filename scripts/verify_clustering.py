import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def run_cluster_verification(tfidf_matrix, k_values=[20, 35, 55, 75], svd_components=500, sample_size=5000, random_state=42):
    print(f"\n{'='*40}\nSTARTING CLUSTER VERIFICATION\n{'='*40}")
    
    # 1. Prepare the SVD Matrix
    print(f"[INFO] Reducing TF-IDF from {tfidf_matrix.shape[1]} to {svd_components} components using SVD...")
    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    
    # 2. Storage for metrics
    wcss_raw, wcss_svd = [], []
    sil_raw, sil_svd = [], []
    
    # 3. Loop through k values and evaluate both
    for k in tqdm(k_values, desc="Evaluating K values"):
        # --- RAW TF-IDF PIPELINE ---
        kmeans_raw = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_raw = kmeans_raw.fit_predict(tfidf_matrix)
        
        wcss_raw.append(kmeans_raw.inertia_)
        # We use a sample size because calculating Silhouette on 84k items takes too long
        sil_raw.append(silhouette_score(tfidf_matrix, labels_raw, metric='cosine', sample_size=sample_size, random_state=random_state))
        
        # --- SVD + K-MEANS PIPELINE ---
        kmeans_svd = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_svd = kmeans_svd.fit_predict(svd_matrix)
        
        wcss_svd.append(kmeans_svd.inertia_)
        sil_svd.append(silhouette_score(svd_matrix, labels_svd, metric='cosine', sample_size=sample_size, random_state=random_state))

    # 4. Plotting the results
    _plot_verification_results(k_values, wcss_raw, wcss_svd, sil_raw, sil_svd, svd_components)


def _plot_verification_results(k_values, wcss_raw, wcss_svd, sil_raw, sil_svd, svd_components):
    # ==========================================
    # FIGURE 1: WCSS (Inertia) Elbow Method (Single Y-Axis)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_title('WCSS (Inertia) Comparison')
    ax1.set_ylabel('WCSS')
    
    # Plot BOTH lines on the exact same axis
    ax1.plot(k_values, wcss_raw, color='tab:blue', marker='o', linewidth=2, label='Raw TF-IDF')
    ax1.plot(k_values, wcss_svd, color='tab:red', marker='s', linewidth=2, label=f'SVD {svd_components} + K-Means')
    
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('wcss_comparison.png', dpi=300, bbox_inches='tight')
    print("[INFO] WCSS plot saved to 'wcss_comparison.png'")
    
    plt.show()
    plt.close(fig1)

    # ==========================================
    # FIGURE 2: Silhouette Scores
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    ax2.plot(k_values, sil_raw, color='tab:blue', marker='o', linewidth=2, label='Raw TF-IDF')
    ax2.plot(k_values, sil_svd, color='tab:red', marker='s', linewidth=2, label=f'SVD {svd_components} + K-Means')
    
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score (Cosine)')
    ax2.set_title('Silhouette Score Comparison')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('silhouette_comparison.png', dpi=300, bbox_inches='tight')
    print("[INFO] Silhouette plot saved to 'silhouette_comparison.png'")
    
    plt.show()
    plt.close(fig2)