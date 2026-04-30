import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_f01_comparison(p_values, kmeans_scores, title_prefix, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, kmeans_scores['top_1'], marker='o', linestyle='-', color='darkred', label='Our Code: Top-1 Cluster')
    plt.plot(p_values, kmeans_scores['top_5'], marker='v', linestyle='-', color='red', label='Our Code: Top-5 Clusters')
    plt.plot(p_values, kmeans_scores['top_10'], marker='*', linestyle='-', color='orange', label='Our Code: Top-10 Clusters')
    plt.plot(p_values, kmeans_scores['top_all'], marker='D', linestyle='-', color='blue', label='Our Code: Top-all (Avg)')
    
    plt.xlabel('p (Fraction of hidden test set)')
    plt.ylabel('F0.1 Score')

    # Have it run between 0 and 1
    plt.ylim(0, 1)

    plt.title(f'Performance Comparison: Context-Aware Music Recommendation ({title_prefix})')
    plt.tight_layout()
    plt.xticks(p_values)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Save silently
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph successfully saved as '{save_path}'")
        plt.show()
    
    # Close the plot so it doesn't consume memory or freeze the script
    plt.close()

def plot_cluster_distribution(df, cluster_col, save_path=None):
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
    plt.xticks(rotation=45) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Distribution graph saved successfully to: {save_path}")
        plt.show()
        
    plt.close()