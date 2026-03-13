import os
import numpy as np
# from pyclustering.cluster import proclus
from clustering.base_clustering import BaseClusteringAlgorithm

class ProclusClustering(BaseClusteringAlgorithm):
    def __init__(self, k=30, l=5, random_state=42):
        # Create a dynamic string based on the parameters
        config_str = f"k{k}_l{l}"
        
        # Pass the name and config to the base class to create the dynamic folder
        super().__init__(algo_name="PROCLUS", config_name=config_str)
        
        self.k = k
        self.l = l # Average number of dimensions for a cluster
        self.random_state = random_state
        
        # State variables to store data for the report
        self.unique_texts_count = 0
        self.cluster_col = f'proclus_cluster_{self.k}'
        self.sanity_check_text = ""

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        self.unique_texts_count = len(unique_texts)
        print(f"\nApplying PROCLUS clustering... on {self.unique_texts_count:,} unique contexts with k={self.k}, l={self.l}")

        # Convert matrix to a dense list of lists
        # WARNING: This will consume a massive amount of RAM on large matrices!
        print("[WARNING] Converting sparse matrix to dense list of lists. Watch your RAM!")
        dense_data = tfidf_matrix.toarray().tolist()

        # Random medoid initialization
        np.random.seed(self.random_state)
        initial_medoids = np.random.choice(len(dense_data), self.k, replace=False).tolist()

        # Init and run
        print("Fitting PROCLUS model... (This might take a while)")
        proclus_instance = proclus(dense_data, initial_medoids, self.l)
        proclus_instance.process()

        # Extract clusters 
        clusters = proclus_instance.get_clusters()
        
        # Free up memory immediately since we have our clusters!
        del dense_data 

        # Convert to same data type as kmeans output
        cluster_labels = np.full(len(unique_texts), -1) # Default to -1 for noise/outliers
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                cluster_labels[idx] = cluster_id
        
        # Zip unique texts with their cluster labels into a DICT (fixed mapping bug)
        cluster_mapping = dict(zip(unique_texts, cluster_labels))

        # Broadcast the cluster back to a new column in the original dataframe
        df[self.cluster_col] = df['expanded_features'].map(cluster_mapping)
        print(f"Added '{self.cluster_col}' labels to the DataFrame.")

        # Generate the sanity check
        self._generate_sanity_check(df)

        return df, self.cluster_col

    def _generate_sanity_check(self, df):
        """Builds the sanity check text and prints it to the console."""
        lines = []
        lines.append(f"\n{'='*30} HIGH-VARIETY CLUSTER SAMPLES {'='*30}")
        
        valid_clusters = df[self.cluster_col].dropna().unique()
        
        # Randomly sample up to 3 clusters (excluding the -1 noise cluster if possible)
        valid_clusters = [c for c in valid_clusters if c != -1]
        n_samples = min(3, len(valid_clusters))
        
        if n_samples == 0:
            self.sanity_check_text = "No valid clusters found for sanity check."
            return

        sampled_ids = np.random.choice(valid_clusters, n_samples, replace=False)

        for c_id in sorted(sampled_ids):
            cluster_df = df[df[self.cluster_col] == c_id]
            all_unique_playlists = cluster_df['playlistname'].unique()
            lines.append(f"\n[Cluster {int(c_id)}] ({len(all_unique_playlists):,} Unique Contexts)")
            
            unique_songs = cluster_df['trackname'].unique()
            np.random.shuffle(unique_songs)
            
            for song in unique_songs[:5]:
                associated_playlists = cluster_df[cluster_df['trackname'] == song]['playlistname'].unique()
                playlist_str = ", ".join(str(p) for p in associated_playlists[:4])
                
                if len(associated_playlists) > 4:
                    playlist_str += f" (+{len(associated_playlists)-4} more)"
                    
                lines.append(f"  🎵 {str(song)[:30]:<32} | Contexts: {playlist_str}")
                
        lines.append(f"{'='*90}")
        
        self.sanity_check_text = "\n".join(lines)
        print(self.sanity_check_text)

    def create_report(self):
        """Generates the isolated text report for PROCLUS."""
        print(f"\nGenerating report for {self.algo_name}...")
        
        report_path = os.path.join(self.report_dir, "run_summary.txt")
        
        with open(report_path, "w") as f:
            f.write(f"--- {self.algo_name} Clustering Report ---\n")
            f.write(f"Target Clusters (k)    : {self.k}\n")
            f.write(f"Subspace Dimensions (l): {self.l}\n")
            f.write(f"random_state           : {self.random_state}\n")
            f.write(f"Total Contexts         : {self.unique_texts_count:,}\n")
            f.write(f"----------------------------------------\n")
            f.write(self.sanity_check_text)
            
        print(f"Report successfully saved to '{self.report_dir}/'")