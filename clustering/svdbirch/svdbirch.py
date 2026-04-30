import os
import numpy as np
from sklearn.cluster import Birch
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from interface.base_algorithm import BaseAlgorithm

class SVDBirchClustering(BaseAlgorithm):
    def __init__(self, n_components=500, threshold=0.5, branching_factor=50, n_clusters=55, batch_size=1000, random_state=42):
        # Create a dynamic string based on SVD, Birch, and batch parameters
        config_str = f"svd{n_components}_thresh{threshold}_bf{branching_factor}_k{n_clusters}_batch{batch_size}"
        
        super().__init__(algo_name="SVDBirch", config_name=config_str)
        
        self.n_components = n_components
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        
        # State variables
        self.unique_texts_count = 0
        self.explained_variance = 0.0
        self.cluster_labels = None
        self.actual_clusters_found = 0
        self.cluster_col = f'svd{self.n_components}_birch_{self.n_clusters}'
        self.sanity_check_text = ""

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        self.unique_texts_count = len(unique_texts)
        
        # 1. Dimensionality Reduction (SVD)
        print(f"\n[INFO] Applying TruncatedSVD to reduce features to {self.n_components}...")
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        
        self.explained_variance = svd.explained_variance_ratio_.sum()
        print(f"[INFO] SVD Explained Variance: {self.explained_variance:.2%} of original information retained.")

        # 2. Batched Birch Clustering
        print(f"\n[INFO] Applying batched BIRCH clustering on {self.unique_texts_count:,} unique contexts...")
        birch = Birch(
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            n_clusters=self.n_clusters
        )

        n_samples = reduced_matrix.shape[0]

        # Partial Fit using batches
        for i in tqdm(range(0, n_samples, self.batch_size), desc="Partial Fitting BIRCH"):
            batch = reduced_matrix[i : i + self.batch_size]
            birch.partial_fit(batch)

        print("[INFO] Predicting labels...")
        self.cluster_labels = birch.predict(reduced_matrix)
        self.actual_clusters_found = len(set(self.cluster_labels))
        
        print(f"[INFO] Assigned cluster labels for {len(self.cluster_labels):,} unique contexts.")

        # Create mapping dictionary
        cluster_mapping = {
            text: label for text, label in tqdm(zip(unique_texts, self.cluster_labels), 
            total=len(unique_texts), desc="Zipping labels")
        }

        # Broadcast to DataFrame
        print(f"[INFO] Broadcasting labels to {len(df):,} rows...")
        tqdm.pandas(desc="Mapping Clusters")
        df[self.cluster_col] = df['expanded_features'].progress_map(cluster_mapping.get)
        
        print(f"[INFO] Added '{self.cluster_col}' labels to the DataFrame.")

        # Generate the sanity check
        self._generate_sanity_check(df)
        
        return df, self.cluster_col

    def _generate_sanity_check(self, df):
        """Builds the sanity check text and prints it to the console."""
        lines = []
        lines.append(f"\n{'='*30} HIGH-VARIETY CLUSTER SAMPLES (SVD+BIRCH) {'='*30}")
        
        valid_clusters = df[self.cluster_col].dropna().unique()
        
        n_samples = min(3, len(valid_clusters))
        if n_samples == 0:
            self.sanity_check_text = "No clusters found for sanity check."
            return

        np.random.seed(self.random_state)
        sampled_ids = np.random.choice(valid_clusters, n_samples, replace=False)

        for c_id in sorted(sampled_ids):
            cluster_df = df[df[self.cluster_col] == c_id]
            all_unique_playlists = cluster_df['playlistname'].unique()
            lines.append(f"\n[Cluster {int(c_id)}] ({len(all_unique_playlists):,} Unique Contexts)")
            
            unique_songs = np.asarray(cluster_df['trackname'].unique(), dtype=object)
            np.random.shuffle(unique_songs)
            
            for song in unique_songs[:5]:
                associated_playlists = cluster_df[cluster_df['trackname'] == song]['playlistname'].unique()
                playlist_str = ", ".join(str(p) for p in associated_playlists[:4])
                
                if len(associated_playlists) > 4:
                    playlist_str += f" (+{len(associated_playlists)-4} more)"
                    
                lines.append(f"  🎵 {str(song)[:30]:<32} | Contexts: {playlist_str}")
                
        lines.append(f"{'='*98}")
        
        self.sanity_check_text = "\n".join(lines)
        print(self.sanity_check_text)

    def create_report(self):
        """Generates the isolated text report for SVD+Birch."""
        print(f"\nGenerating report for {self.algo_name}...")
        
        report_path = os.path.join(self.report_dir, "run_summary.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"--- {self.algo_name} Clustering Report ---\n")
            f.write(f"Configuration: {self.config_name}\n")
            f.write(f"Total Contexts Clustered: {self.unique_texts_count:,}\n")
            f.write(f"SVD Components: {self.n_components}\n")
            f.write(f"SVD Explained Variance: {self.explained_variance:.2%}\n")
            f.write(f"Birch Threshold: {self.threshold}\n")
            f.write(f"Birch Branching Factor: {self.branching_factor}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Target Clusters (k): {self.n_clusters}\n")
            f.write(f"Actual Clusters Found: {self.actual_clusters_found}\n")
            f.write(self.sanity_check_text)
            
        print(f"[INFO] Report successfully saved to: {report_path}")