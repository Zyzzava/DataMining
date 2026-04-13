import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

from interface.base_algorithm import BaseAlgorithm


class SpectralGraphClustering(BaseAlgorithm):
    def __init__(self, graph, n_clusters=55, graph_config_name="default", random_state=42):
        config_str = f"{graph_config_name}_k{n_clusters}"
        super().__init__(algo_name="Spectral", config_name=config_str)

        self.graph = graph
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_col = f"spectral_cluster_{self.n_clusters}"
        self.partition = None
        self.sanity_check_text = ""

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        print(
            f"\nRunning spectral clustering on {self.graph.number_of_nodes():,} graph nodes (k={self.n_clusters})..."
        )
        self.partition = self._cluster_graph()

        df[self.cluster_col] = df["expanded_features"].map(self.partition)
        print(f"Added '{self.cluster_col}' labels to the DataFrame.")

        self.generate_sanity_check(df)
        return df, self.cluster_col

    def _cluster_graph(self):
        nodes = list(self.graph.nodes())
        if len(nodes) == 0:
            return {}

        if len(nodes) == 1:
            return {nodes[0]: 0}

        # 1. Get the adjacency matrix as a sparse array
        adjacency = nx.to_scipy_sparse_array(self.graph, nodelist=nodes, weight="weight", format="csr")
        
        # 2. FIX: Convert indices to 32-bit to satisfy Scikit-Learn's requirement
        # This resolves: "ValueError: Only sparse matrices with 32-bit integer indices are accepted"
        adjacency.indices = adjacency.indices.astype('int32')
        adjacency.indptr = adjacency.indptr.astype('int32')

        if adjacency.nnz == 0:
            return {node: 0 for node in nodes}

        n_clusters = min(self.n_clusters, len(nodes))
        
        # 3. Initialize Spectral Clustering
        # We use 'arpack' because it is much more stable for disconnected components 
        # (which your graph has, as indicated by the 'Graph is not fully connected' warning)
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="cluster_qr",
            eigen_solver="arpack", 
            random_state=self.random_state,
            n_jobs=-1 # Utilize M4 Pro cores
        )

        print(f"Fitting spectral model on {adjacency.shape[0]} nodes...")
        labels = spectral.fit_predict(adjacency)
        return dict(zip(nodes, labels))

    def generate_sanity_check(self, df):
        lines = [f"\n{'='*30} SPECTRAL COMMUNITY SAMPLES {'='*30}"]
        valid_communities = df[self.cluster_col].dropna().unique()

        n_samples = min(3, len(valid_communities))
        if n_samples > 0:
            sampled_ids = np.random.choice(valid_communities, n_samples, replace=False)
            for c_id in sorted(sampled_ids):
                comm_df = df[df[self.cluster_col] == c_id]
                lines.append(
                    f"\n[Community {int(c_id)}] ({len(comm_df['playlistname'].unique()):,} Unique Contexts)"
                )

                unique_songs = np.asarray(comm_df["trackname"].unique(), dtype=object)
                np.random.shuffle(unique_songs)
                for song in unique_songs[:5]:
                    lines.append(f"  🎵 {str(song)[:30]:<32}")

        self.sanity_check_text = "\n".join(lines)
        print(self.sanity_check_text)

    def create_report(self):
        print(f"\nGenerating report for {self.algo_name}...")

        report_path = os.path.join(self.report_dir, "run_summary.txt")
        with open(report_path, "w") as f:
            f.write(f"Algorithm: {self.algo_name}\n")
            f.write(f"Graph Nodes: {self.graph.number_of_nodes()}\n")
            f.write(f"Graph Edges: {self.graph.number_of_edges()}\n")
            f.write(f"Target Clusters: {self.n_clusters}\n")
            f.write(self.sanity_check_text)