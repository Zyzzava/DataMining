import json
import numpy as np

from evaluation.plot_comparison import plot_f01_comparison

# 1. Load the raw cluster scores from the JSON file
base_path = "clustering/reports/SVDKMeans/svd500_k55_ninit10_maxiter300/" # Adjust the path/folder if needed
json_path = base_path + "raw_cluster_scores.json" # Adjust the path/folder if needed
with open(json_path, "r") as f:
    raw_scores = json.load(f)

# 2. Define the p-values used in the evaluation
p_values = [0.1, 0.3, 0.5, 0.7, 1.0]

# 3. Initialize the dictionary to hold our aggregated scores
kmeans_scores = {
    'top_1': [],
    'top_5': [],
    'top_10': [],
    'top_all': []
}

# 4. Calculate the top-k averages for each p-value
for p in p_values:
    p_str = str(p)
    
    # Extract all valid scores across all clusters for this specific p-value
    scores_for_p = [cluster_data[p_str] for cluster_id, cluster_data in raw_scores.items() if p_str in cluster_data]
    
    # Sort the scores in descending order (highest F0.1 scores first)
    scores_for_p.sort(reverse=True)
    
    # Calculate the means for the top K clusters and append to our lists
    kmeans_scores['top_1'].append(np.mean(scores_for_p[:1]))
    kmeans_scores['top_5'].append(np.mean(scores_for_p[:5]))
    kmeans_scores['top_10'].append(np.mean(scores_for_p[:10]))
    kmeans_scores['top_all'].append(np.mean(scores_for_p))

# 5. Plot using your UPDATED function with the fixed 0-1 y-axis!
plot_f01_comparison(
    p_values=p_values,
    kmeans_scores=kmeans_scores,
    title_prefix="SVDKMeans K=55", # Update this title to whatever you prefer
    save_path=base_path + "f01_comparison.png"
)