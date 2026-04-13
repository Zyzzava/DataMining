import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


class KNNGraph:
    def __init__(self, k_neighbors=10, sim_threshold=0.15):
        self.k = k_neighbors
        self.sim_threshold = sim_threshold
        self.G = nx.Graph()

    def build_graph(self, tfidf_matrix, labels):
        self.G.clear()

        if len(labels) < 2:
            return self.G

        n_neighbors = min(self.k, len(labels))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
        nn.fit(tfidf_matrix)
        distances, indices = nn.kneighbors(tfidf_matrix)

        for i, neighbors_idx in tqdm(enumerate(indices), total=len(indices), desc="Creating Graph Edges"):
            for local_idx, j in enumerate(neighbors_idx):
                if i != j:
                    sim_weight = 1 - distances[i][local_idx]
                    if sim_weight > self.sim_threshold:
                        self.G.add_edge(labels[i], labels[j], weight=float(sim_weight))

        return self.G

    def visualize_improved(self, partition=None, save_path=None):
        if self.G.number_of_nodes() == 0:
            return

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.G, k=0.5, iterations=50, seed=42)

        if partition:
            node_colors = [partition.get(node, 0) for node in self.G.nodes()]
        else:
            node_colors = "#4c78a8"

        nx.draw_networkx_nodes(
            self.G,
            pos,
            node_size=30,
            node_color=node_colors,
            cmap=plt.cm.RdYlBu,
            alpha=0.7,
        )
        nx.draw_networkx_edges(self.G, pos, alpha=0.03, edge_color="black")

        sample_nodes = list(self.G.nodes())[:5]
        labels = {node: (node[:15] + "..") for node in sample_nodes}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=7)

        plt.title(f"Spotify k-NN Graph (Nodes: {self.G.number_of_nodes()})")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        plt.show()