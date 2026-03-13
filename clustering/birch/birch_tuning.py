import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import warnings

# Suppress warnings for when BIRCH creates too few clusters during testing
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = "clustering/birch/birch_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def tune_birch_hyperparameters(tfidf_matrix, k=55, sample_size=15000):
    """
    Tests various combinations of threshold and branching_factor on a subset 
    of data to find the optimal balance of speed, memory, and cluster quality.
    """
    print(f"\n--- Starting BIRCH Hyperparameter Tuning ---")
    print(f"Sampling {sample_size:,} records for faster evaluation...")
    
    # Take a random sample to keep execution time reasonable
    if tfidf_matrix.shape[0] > sample_size:
        X_sample = resample(tfidf_matrix, n_samples=sample_size, random_state=42)
    else:
        X_sample = tfidf_matrix

    # Parameter Grid for TF-IDF
    # TF-IDF distances are usually small, so we test smaller thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    branching_factors = [25, 50, 100]
    
    results = []

    total_tests = len(thresholds) * len(branching_factors)
    current_test = 1

    for t in thresholds:
        for b in branching_factors:
            print(f"Testing {current_test}/{total_tests}: Threshold={t}, Branching={b}...")
            
            # Start timer
            start_time = time.time()
            
            # Initialize BIRCH
            birch = Birch(threshold=t, branching_factor=b, n_clusters=k, copy=False)
            
            try:
                # Fit and predict on the sample
                labels = birch.fit_predict(X_sample)
                fit_time = time.time() - start_time
                
                # Check how many unique clusters were actually formed
                unique_clusters = len(np.unique(labels))
                
                # Calculate Silhouette Score (Quality) - only if we have >1 cluster
                if 1 < unique_clusters <= k:
                    # Using a smaller sample for silhouette to save time
                    sil_score = silhouette_score(X_sample, labels, sample_size=5000)
                else:
                    sil_score = -1.0 # Penalize failed clustering
                    
                results.append({
                    'Threshold': t,
                    'Branching_Factor': b,
                    'Time_Seconds': fit_time,
                    'Clusters_Found': unique_clusters,
                    'Silhouette_Score': sil_score
                })
                
            except Exception as e:
                print(f"  -> Failed: {str(e)}")
                results.append({
                    'Threshold': t,
                    'Branching_Factor': b,
                    'Time_Seconds': np.nan,
                    'Clusters_Found': 0,
                    'Silhouette_Score': -1.0
                })
                
            current_test += 1

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Generate Visualizations
    plot_tuning_results(df_results)
    
    # Save raw data
    df_results.to_csv(f"{OUTPUT_DIR}/birch_tuning_results.csv", index=False)
    print(f"\nTuning complete! Results saved to '{OUTPUT_DIR}'")
    
    # Find and print the best combo
    best_combo = df_results.loc[df_results['Silhouette_Score'].idxmax()]
    print("\n*** RECOMMENDED PARAMETERS ***")
    print(f"Threshold: {best_combo['Threshold']}")
    print(f"Branching Factor: {best_combo['Branching_Factor']}")
    print(f"Estimated Fit Time (on 15k sample): {best_combo['Time_Seconds']:.2f}s")
    print(f"Quality (Silhouette): {best_combo['Silhouette_Score']:.4f}")

def plot_tuning_results(df):
    """ Generates heatmaps for the grid search results. """
    print("Generating heatmaps...")
    
    # Pivot tables for heatmaps
    time_pivot = df.pivot(index='Threshold', columns='Branching_Factor', values='Time_Seconds')
    sil_pivot = df.pivot(index='Threshold', columns='Branching_Factor', values='Silhouette_Score')
    cluster_pivot = df.pivot(index='Threshold', columns='Branching_Factor', values='Clusters_Found')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Time
    sns.heatmap(time_pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[0])
    axes[0].set_title("Execution Time (Seconds) ↓ lower is better")

    # Plot 2: Clusters Found
    sns.heatmap(cluster_pivot, annot=True, fmt=".0f", cmap="Blues", ax=axes[1])
    axes[1].set_title("Clusters Found (Target=55)")

    # Plot 3: Silhouette Score
    # Mask negative scores (failed clusters) for better color scaling
    mask = sil_pivot < 0
    sns.heatmap(sil_pivot, annot=True, fmt=".3f", cmap="Greens", mask=mask, ax=axes[2])
    axes[2].set_title("Cluster Quality (Silhouette) ↑ higher is better")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hyperparameter_heatmaps.png")
    plt.close()