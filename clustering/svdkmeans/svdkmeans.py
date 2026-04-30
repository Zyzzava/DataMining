import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from interface.base_algorithm import BaseAlgorithm

class SVDKMeansClustering(BaseAlgorithm):
    def __init__(self, n_components=500, k=55, n_init=10, max_iter=300, random_state=42):
        # Create a dynamic string based on the SVD and KMeans parameters
        config_str = f"svd{n_components}_k{k}_ninit{n_init}_maxiter{max_iter}"
        
        # Pass the name and config to the base class to create the dynamic folder
        super().__init__(algo_name="SVDKMeans", config_name=config_str)
        
        self.n_components = n_components
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        
        # State variables
        self.unique_texts_count = 0
        self.explained_variance = 0.0
        self.cluster_col = f'svd{self.n_components}_kmeans_{self.k}'
        self.sanity_check_text = ""

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        self.unique_texts_count = len(unique_texts)
        
        # 1. Dimensionality Reduction (SVD)
        print(f"\n[INFO] Applying TruncatedSVD to reduce features to {self.n_components}...")
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        
        # Calculate how much information we kept
        self.explained_variance = svd.explained_variance_ratio_.sum()
        print(f"[INFO] SVD Explained Variance: {self.explained_variance:.2%} of original information retained.")

        # 2. KMeans Clustering
        print(f"[INFO] Applying KMeans clustering on reduced matrix with k={self.k}...")
        kmeans = KMeans(
            n_clusters=self.k, 
            random_state=self.random_state, 
            n_init=self.n_init,
            max_iter=self.max_iter
        )
        cluster_labels = kmeans.fit_predict(reduced_matrix)

        # Create mapping and broadcast to DataFrame
        cluster_mapping = dict(zip(unique_texts, cluster_labels))
        df[self.cluster_col] = df['expanded_features'].map(cluster_mapping)
        
        print(f"[INFO] Added '{self.cluster_col}' labels to the DataFrame.")

        # Generate the sanity check
        self._generate_sanity_check(df)
        
        return df, self.cluster_col

    def _generate_sanity_check(self, df):
        """Builds the sanity check text and prints it to the console."""
        lines = []
        lines.append(f"\n{'='*30} HIGH-VARIETY CLUSTER SAMPLES (SVD+KMEANS) {'='*30}")
        
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
                
        lines.append(f"{'='*98}")
        
        # Store as a single string for the text file, and print to console for live viewing
        self.sanity_check_text = "\n".join(lines)
        print(self.sanity_check_text)

    def create_report(self):
        """Generates the isolated text report for SVD+KMeans."""
        print(f"\nGenerating report for {self.algo_name}...")
        
        report_path = os.path.join(self.report_dir, "run_summary.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Algorithm: {self.algo_name}\n")
            f.write(f"Configuration: {self.config_name}\n")
            f.write(f"Total Contexts Clustered: {self.unique_texts_count:,}\n")
            f.write(f"SVD Components: {self.n_components}\n")
            f.write(f"SVD Explained Variance: {self.explained_variance:.2%}\n")
            f.write(f"K-Means Clusters (k): {self.k}\n")
            f.write(self.sanity_check_text)
            
        print(f"[INFO] Report successfully saved to: {report_path}")