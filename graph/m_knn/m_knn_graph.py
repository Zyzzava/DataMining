import os
import pickle
import networkx as nx

class M_KNN:
    def __init__(self, digraph: nx.DiGraph):
        self.digraph = digraph

    def build_graph(self, force_rebuild=True):
        file_path = "graph/m_knn/saved_graphs/mutual_knn_graph.pkl"
        
        if os.path.exists(file_path) and not force_rebuild:
            print("Loading mutual k-NN NetworkX graph from .pkl file...")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        print("Building mutual k-NN using NetworkX Intersection...")
        
        reversed_digraph = self.digraph.reverse(copy=True)
        mutual_directed = nx.intersection(self.digraph, reversed_digraph)
        mutual_knn_nx_graph = mutual_directed.to_undirected()
        
        print("Saving new NetworkX graph to .pkl file...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(mutual_knn_nx_graph, f)
        print("Graph saved successfully for future runs!")
        
        return mutual_knn_nx_graph