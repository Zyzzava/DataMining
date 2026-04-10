import os
import numpy as np
from sklearn.cluster import KMeans
from interface.base_algorithm import BaseAlgorithm

class KMeansClustering(BaseAlgorithm):
    def __init__(self, k=30, n_init=10, max_iter=300, random_state=42):
        # Create a dynamic string based on the parameters
        config_str = f"k{k}_ninit{n_init}_maxiter{max_iter}"
        
        # Pass the name and config to the base class to create the dynamic folder
        super().__init__(algo_name="KMeans", config_name=config_str)
        
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        
        # State variables to store data for the report
        self.unique_texts_count = 0
        self.cluster_col = f'k-means_cluster_{self.k}'
        self.sanity_check_text = ""

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        self.unique_texts_count = len(unique_texts)
        print(f"\nApplying KMeans clustering... on {self.unique_texts_count:,} unique contexts with k={self.k}")

        # Fit the model
        kmeans = KMeans(
            n_clusters=self.k, 
            random_state=self.random_state, 
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        # Create mapping and broadcast to DataFrame
        cluster_mapping = dict(zip(unique_texts, cluster_labels))
        df[self.cluster_col] = df['expanded_features'].map(cluster_mapping)
        
        print(f"Added '{self.cluster_col}' labels to the DataFrame.")

        # Generate the sanity check while we still have access to the modified DataFrame
        self._generate_sanity_check(df)
        
        return df, self.cluster_col

    def _generate_sanity_check(self, df):
        """Builds the sanity check text and prints it to the console."""
        lines = []
        lines.append(f"\n{'='*30} HIGH-VARIETY CLUSTER SAMPLES {'='*30}")
        
        valid_clusters = df[self.cluster_col].dropna().unique()
        
        # Randomly sample up to 3 clusters
        n_samples = min(3, len(valid_clusters))
        if n_samples == 0:
            self.sanity_check_text = "No clusters found for sanity check."
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
        
        # Store as a single string for the text file, and print to console for live viewing
        self.sanity_check_text = "\n".join(lines)
        print(self.sanity_check_text)

    def create_report(self):
        """Generates the isolated text report for KMeans."""
        print(f"\nGenerating report for {self.algo_name}...")
        
        report_path = os.path.join(self.report_dir, "run_summary.txt")