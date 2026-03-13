import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_f01_comparison(p_values, kmeans_scores, title_prefix):
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
    file_name = f"{title_prefix.replace(' ', '_').lower()}_f01_comparison.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph successfully saved as '{file_name}'")
    plt.show()


def plot_cluster_distribution(df, cluster_col):
    print(f"\n{'='*30}\nPLOTTING CLUSTER DISTRIBUTION\n{'='*30}")

    cluster_counts = df[cluster_col].dropna().value_counts()

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=cluster_counts.index.astype(int), 
        y=cluster_counts.values, 
        palette='viridis', 
        hue=cluster_counts.index.astype(int), 
        legend=False
    )

    # 5. Formatting the Graph
    plt.title(f'Cluster Size Distribution ({cluster_col})', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster ID (Ranked Largest to Smallest)', fontsize=12)
    plt.ylabel('Number of Playlists/Tracks', fontsize=12)
    
    # Rotate the x-axis labels in case you have 50+ clusters and they overlap
    plt.xticks(rotation=45) 
    
    # Add a subtle grid line on the y-axis to make it easy to read the heights
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 6. Save and Show
    output_dir = "evaluation/graphs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"distribution_{cluster_col}.png")
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"-> Distribution graph saved successfully to: {file_path}")
    
    plt.show()