from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd
import numpy as np

def calculate_wcss(df_column, sample_frac=0.05):
    # 1. Extract strictly UNIQUE expanded playlists
    unique_playlists = df_column.dropna().unique()
    print(f"Total unique playlists to process: {len(unique_playlists):,}")

    # 2. Optional: Sub-sample from the unique list if it's still too large
    if sample_frac < 1.0:
        sample_size = int(len(unique_playlists) * sample_frac)
        unique_playlists = np.random.choice(unique_playlists, sample_size, replace=False)
        print(f"Sub-sampled to {len(unique_playlists):,} unique playlists for WCSS.")

    # 3. Create the TF-IDF Matrix (on unique strings only)
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(unique_playlists)

    wcss = []
    k_range = range(2, 100) 

    # 4. KMeans Loop
    for k in tqdm(k_range, desc="Calculating WCSS (Unique)"):
        kmeans = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init='auto'
        )
        kmeans.fit(tfidf_matrix)
        wcss.append(kmeans.inertia_)

    # 5. Numerical Selection (De-trending)
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
