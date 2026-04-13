import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from interface.base_algorithm import BaseAlgorithm


class LouvainClustering(BaseAlgorithm):
    def __init__(self, graph, graph_config_name="default"):
        super().__init__(algo_name="Louvain", config_name=graph_config_name)

        self.graph = graph
        self.cluster_col = "louvain_community_labels"
        self.partition = None
        self.sanity_check_text = ""

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        print(f"\nRunning Louvain community detection on {self.graph.number_of_nodes():,} graph nodes...")
        self.partition = self.detect_communities()

        df[self.cluster_col] = df["expanded_features"].map(self.partition)
        print(f"Added '{self.cluster_col}' labels to the DataFrame.")

        self.generate_sanity_check(df)
        return df, self.cluster_col

    def detect_communities(self):
        try:
            import community as community_louvain

            return community_louvain.best_partition(self.graph)
        except ImportError:
            print("Error: 'python-louvain' not installed. Falling back to dummy partition.")
        except Exception as exc:
            print(f"Error while running Louvain clustering: {exc}. Falling back to a single community.")

        return {node: 0 for node in self.graph.nodes()}

    def generate_sanity_check(self, df):
        lines = [f"\n{'='*30} LOUVAIN COMMUNITY SAMPLES {'='*30}"]
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
            f.write(self.sanity_check_text)