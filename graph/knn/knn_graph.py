import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from interface.base_algorithm import BaseAlgorithm

class KNNGraph(BaseAlgorithm):
    def __init__(self, k_neighbors=10, sim_threshold=0.15):
        # Create dynamic string for the folder structure
        config_str = f"k{k_neighbors}_sim{sim_threshold}"
        super().__init__(algo_name="KNNGraph", config_name=config_str)
        
        self.k = k_neighbors
        self.sim_threshold = sim_threshold
        self.G = nx.Graph()
        
        # Interface alignment
        self.cluster_col = "graph_community_labels"
        self.sanity_check_text = ""
        self.partition = None

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        """Matches the KMeans interface: Runs graph mining and maps to DataFrame."""
        print(f"\nBuilding k-NN Graph on {len(unique_texts):,} contexts (k={self.k})...")
        
        # 1. Build Graph
        self._build_graph(tfidf_matrix, unique_texts)
        
        # 2. Graph Mining: Louvain Community Detection
        print("[INFO] Detecting communities (Louvain Mining)...")
        self.partition = self.detect_communities()
        
        # 3. Map to DataFrame
        df[self.cluster_col] = df['expanded_features'].map(self.partition)
        print(f"Added '{self.cluster_col}' labels to the DataFrame.")

        # 4. Generate Sanity Check
        self.generate_sanity_check(df)
        
        return df, self.cluster_col

    def build_graph(self, tfidf_matrix, labels):
        nn = NearestNeighbors(n_neighbors=self.k, metric='cosine', n_jobs=-1)
        nn.fit(tfidf_matrix)
        distances, indices = nn.kneighbors(tfidf_matrix)

        for i, neighbors_idx in tqdm(enumerate(indices), total=len(indices), desc="Creating Graph Edges"):
            for local_idx, j in enumerate(neighbors_idx):
                if i != j:
                    sim_weight = 1 - distances[i][local_idx]
                    if sim_weight > self.sim_threshold: 
                        self.G.add_edge(labels[i], labels[j], weight=float(sim_weight))

    def detect_communities(self):
        try:
            import community as community_louvain
            return community_louvain.best_partition(self.G)
        except ImportError:
            print("Error: 'python-louvain' not installed. Falling back to dummy partition.")
            return {node: 0 for node in self.G.nodes()}

    def generate_sanity_check(self, df):
        """Matches KMeans style console output for the graph clusters."""
        lines = [f"\n{'='*30} GRAPH COMMUNITY SAMPLES {'='*30}"]
        valid_communities = df[self.cluster_col].dropna().unique()
        
        n_samples = min(3, len(valid_communities))
        if n_samples > 0:
            sampled_ids = np.random.choice(valid_communities, n_samples, replace=False)
            for c_id in sorted(sampled_ids):
                comm_df = df[df[self.cluster_col] == c_id]
                lines.append(f"\n[Community {int(c_id)}] ({len(comm_df['playlistname'].unique()):,} Unique Contexts)")
                
                unique_songs = comm_df['trackname'].unique()
                np.random.shuffle(unique_songs)
                for song in unique_songs[:5]:
                    lines.append(f"  🎵 {str(song)[:30]:<32}")
        
        self.sanity_check_text = "\n".join(lines)
        print(self.sanity_check_text)

    def create_report(self):
        """Generates the visualization and saves summary text."""
        print(f"\nGenerating report and visualization for {self.algo_name}...")
        
        # Save Visualization
        self.visualize_improved(save_path=os.path.join(self.report_dir, "graph_visualization.png"))
        
        # Save Run Summary
        report_path = os.path.join(self.report_dir, "run_summary.txt")
        with open(report_path, "w") as f:
            f.write(f"Algorithm: {self.algo_name}\n")
            f.write(f"Nodes: {self.G.number_of_nodes()}\n")
            f.write(f"Edges: {self.G.number_of_edges()}\n")
            f.write(self.sanity_check_text)

    def visualize_improved(self, save_path=None):
        if not self.partition:
            return
            
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=42)
        
        nx.draw_networkx_nodes(self.G, pos, node_size=30, 
                               node_color=list(self.partition.values()), 
                               cmap=plt.cm.RdYlBu, alpha=0.7)
        nx.draw_networkx_edges(self.G, pos, alpha=0.03, edge_color='black')
        
        # Small label sample
        sample_nodes = list(self.G.nodes())[:5]
        labels = {node: (node[:15] + '..') for node in sample_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=7)

        plt.title(f"Spotify k-NN Communities (Nodes: {self.G.number_of_nodes()})")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        plt.show()