import matplotlib.pyplot as plt
import os

def plot_f01_comparison(p_values, kmeans_scores):
    output_dir = "evaluation/graphs"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, kmeans_scores['top_1'], marker='o', linestyle='-', color='darkred', label='Our Code: Top-1 Cluster')
    plt.plot(p_values, kmeans_scores['top_5'], marker='v', linestyle='-', color='red', label='Our Code: Top-5 Clusters')
    plt.plot(p_values, kmeans_scores['top_10'], marker='*', linestyle='-', color='orange', label='Our Code: Top-10 Clusters')
    plt.plot(p_values, kmeans_scores['top_all'], marker='D', linestyle='-', color='blue', label='Our Code: Top-all (Avg)')
    plt.xlabel('p (Fraction of hidden test set)')
    plt.ylabel('F0.1 Score')
    plt.title('Performance Comparison: Context-Aware Music Recommendation (K-Means)')
    plt.xticks(p_values)
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save and Show
    file_path = os.path.join(output_dir, "f01_replication_graph.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print("\nGraph successfully saved as 'f01_replication_graph.png'")
    plt.show()