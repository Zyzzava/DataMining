import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. Load and Prepare Spotify Data
# ==========================================
print("Loading Spotify dataset...")
df = pd.read_parquet('data/spotify_fully_processed.parquet')

print("Filtering for contextual features...")
contextual_mask = df['is_contextual'] == True
unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()

print(f"Creating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, max_features=5678)
tfidf_matrix = vectorizer.fit_transform(unique_texts)

X_full = tfidf_matrix

# Cap rows to prevent memory crashes during 320+ iterations
MAX_ROWS = 5000 
num_rows = X_full.shape[0]

if num_rows > MAX_ROWS:
    print(f"Downsampling to {MAX_ROWS} rows for Spectral K-analysis...")
    np.random.seed(42)
    indices = np.random.choice(num_rows, MAX_ROWS, replace=False)
    X = X_full[indices].toarray() 
else:
    X = X_full.toarray()

print(f"Data shape ready for clustering: {X.shape}")

# ==========================================
# 2. The Eigengap Heuristic (Mathematical)
# ==========================================
print("Computing Eigengap heuristic...")

connectivity = kneighbors_graph(X, n_neighbors=15, mode='connectivity', include_self=True)
affinity_matrix = 0.5 * (connectivity + connectivity.T)
L = laplacian(affinity_matrix, normed=True)

# Extract first 330 eigenvalues to safely find gaps up to k=322
eigenvalues, _ = eigh(L.toarray())
eigenvalues = np.sort(eigenvalues)[:330]

gaps = np.diff(eigenvalues)
optimal_k_eigen = np.argmax(gaps[1:]) + 2

# ==========================================
# 3. Metric Sweeping & File Saving
# ==========================================
print("Running parameter sweep for K...")

k_range = range(675, 726, 1) 
silhouette_scores = []
db_scores = []

results_filename = "spectral_k_results.txt"

# Open the file and write headers
with open(results_filename, "w") as f:
    f.write("K_Value\tSilhouette_Score\tDavies_Bouldin_Score\n")
    f.write("-" * 55 + "\n")

    for k in k_range:
        print(f"  Testing k={k}...")
        sc = SpectralClustering(
            n_clusters=k, 
            affinity='nearest_neighbors', 
            n_neighbors=15, 
            assign_labels='cluster_qr', 
            random_state=42
        )
        labels = sc.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        
        silhouette_scores.append(sil_score)
        db_scores.append(db_score)
        
        # Write to file immediately and flush buffer so data isn't lost if interrupted
        f.write(f"{k}\t{sil_score:.6f}\t{db_score:.6f}\n")
        f.flush() 

optimal_k_sil = k_range[np.argmax(silhouette_scores)]

# Append the final conclusion to the text file
with open(results_filename, "a") as f:
    f.write("-" * 55 + "\n")
    f.write(f"FINAL CONCLUSION:\n")
    f.write(f"Optimal K (Eigengap Method): {optimal_k_eigen}\n")
    f.write(f"Optimal K (Highest Silhouette): {optimal_k_sil}\n")

print(f"\nSaved all results to {results_filename}")

# ==========================================
# 4. Plotting the Results
# ==========================================
print("Generating plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Eigengap
ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', markersize=2, linestyle='-', color='b')
ax1.set_title(f'Eigengap Heuristic (Suggested k={optimal_k_eigen})')
ax1.set_xlabel('Index of Eigenvalue (k)')
ax1.set_ylabel('Eigenvalue')
ax1.axvline(x=optimal_k_eigen, color='r', linestyle='--', label=f'Max Gap at k={optimal_k_eigen}')
ax1.legend()
ax1.grid(True)

# Plot 2: Silhouette & Davies-Bouldin
color = 'tab:blue'
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(k_range, silhouette_scores, marker='.', markersize=4, color=color, label='Silhouette (Higher is better)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axvline(x=optimal_k_sil, color='g', linestyle='--', label=f'Best Silhouette (k={optimal_k_sil})')

# Create a twin axis for Davies-Bouldin
ax3 = ax2.twinx()  
color = 'tab:orange'
ax3.set_ylabel('Davies-Bouldin Score', color=color)  
ax3.plot(k_range, db_scores, marker='.', markersize=4, color=color, label='Davies-Bouldin (Lower is better)')
ax3.tick_params(axis='y', labelcolor=color)

# Combine legends for ax2 and ax3
lines_1, labels_1 = ax2.get_legend_handles_labels()
lines_2, labels_2 = ax3.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

plt.title('Empirical Metrics Sweep (Up to k=322)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('spectral_k_analysis_spotify.png', dpi=300)
print("Saved plot to 'spectral_k_analysis_spotify.png'")

print(f"\n--- Final Results ---")
print(f"Suggested k (Eigengap): {optimal_k_eigen}")
print(f"Suggested k (Silhouette): {optimal_k_sil}")