import os
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np

def calculate_wcss(df_column, tfidf_matrix, sample_frac=0.05):
    print(f"Calculating WCSS for matrix of shape: {tfidf_matrix.shape}")

    wcss = []
    k_range = range(2, 100) 

    # KMeans Loop
    for k in tqdm(k_range, desc="Calculating WCSS"):
        kmeans = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init='auto'
        )
        kmeans.fit(tfidf_matrix)
        wcss.append(kmeans.inertia_)

    # Numerical Selection (De-trending)
    wcss_diff = np.diff(wcss)
    avg_delta = np.mean(wcss_diff)
    std_delta = np.std(wcss_diff)

    threshold = avg_delta - std_delta

    optimal_k = 2
    for i, delta in enumerate(wcss_diff):
        if delta < threshold:
            optimal_k = k_range[i+1]

    print(f"\nOptimal clusters found: {optimal_k}")
    return optimal_k, wcss, wcss_diff, avg_delta, std_delta, k_range

def graph_wcss(wcss, k_range, title_suffix=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o')
    plt.title(f'WCSS vs. Number of Clusters (k) [Sampled] {title_suffix}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.xticks(list(k_range)[::10]) 
    plt.grid(True)
    plt.savefig(f'WCSS/wcss_graph{title_suffix}.png')

def graph_delta_wcss(wcss_delta, k_range, avg_delta, std_delta, threshold, title_suffix=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    
    # x-axis for diffs is shifted by 1 (e.g., the diff between k=2 and k=3 is plotted at k=3)
    k_range_diff = list(k_range)[1:] 
    
    # Plot the actual Delta WCSS line
    plt.plot(k_range_diff, wcss_delta, marker='o', label='Delta WCSS', color='b')
    
    # Plot the Statistical Lines
    plt.axhline(avg_delta, color='r', linestyle='--', label=f'Mean ({avg_delta:.0f})')
    plt.axhline(threshold, color='g', linestyle='-.', label=f'Threshold [Mean - Std] ({threshold:.0f})')
    plt.axhline(avg_delta + std_delta, color='orange', linestyle=':', label=f'Mean + Std ({avg_delta + std_delta:.0f})')

    plt.title(f'Delta WCSS vs. Number of Clusters (k) {title_suffix}')
    plt.xlabel('Number of Clusters (k) [Causing the drop]')
    plt.ylabel('Delta WCSS')
    plt.xticks(k_range_diff[::5]) # Show every 5th tick for clean reading
    plt.legend()
    plt.grid(True)
    
    import os
    if not os.path.exists('WCSS'):
        os.makedirs('WCSS')
    plt.savefig(f'WCSS/delta_wcss_graph{title_suffix}.png')
    plt.show()

def calculate_and_graph_wcss(tfidf_matrix):
    """Calculates and graphs WCSS for optimal k discovery."""
    if os.path.exists('WCSS') and os.listdir('WCSS'):
        print("\nWCSS graphs already exist. Skipping WCSS calculation and graphing.")
        return

    print("\nCalculating WCSS to determine optimal number of clusters...")

    df_column = 'expanded_features'
    sample_frac = 1.0

    optimal_k, wcss, wcss_delta, avg_delta, std_delta, k_range = calculate_wcss(df_column, tfidf_matrix, sample_frac=sample_frac)
    
    if not os.path.exists('WCSS'):
        os.makedirs('WCSS')
        
    graph_wcss(wcss, k_range, title_suffix=f"(Optimal k={optimal_k})")
    threshold = avg_delta - std_delta
    graph_delta_wcss(wcss_delta, k_range, avg_delta, std_delta, threshold, title_suffix=f"(Delta WCSS)")
    return optimal_k