import os
import numpy as np
from sklearn.cluster import SpectralBiclustering
from interface.base_algorithm import BaseAlgorithm

class BiclusteringAlgorithm(BaseAlgorithm):
    def __init__(self, n_row_clusters=55, n_column_clusters=10, random_state=42):
        # Create a dynamic string based on the parameters
        config_str = f"kRows{n_row_clusters}_kCols{n_column_clusters}"
        
        # Pass the name and config to the base class to create the dynamic folder
        super().__init__(algo_name="SpectralBiclustering", config_name=config_str)
        
        self.n_row_clusters = n_row_clusters
        self.n_column_clusters = n_column_clusters
        self.random_state = random_state
        
        # State variables
        self.unique_texts_count = 0
        self.cluster_col = f'bicluster_row_{self.n_row_clusters}'
        self.sanity_check_text = ""

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        self.unique_texts_count = len(unique_texts)
        print(f"\nApplying Spectral Biclustering on {self.unique_texts_count:,} unique contexts...")
        print(f"Targeting {self.n_row_clusters} row clusters and {self.n_column_clusters} column clusters.")

        # Fit the model
        # We explicitly set method='bistochastic' because the TF-IDF matrix is a sparse matrix
        bicluster = SpectralBiclustering(
            n_clusters=(self.n_row_clusters, self.n_column_clusters),
            random_state=self.random_state,
            svd_method='arpack',
            method='bistochastic' 
        )
        
        # Fit on the sparse TF-IDF matrix
        bicluster.fit(tfidf_matrix)

        # In biclustering, we care primarily about the row labels for the playlists
        row_cluster_labels = bicluster.row_labels_

        # Create mapping and broadcast to DataFrame
        cluster_mapping = dict(zip(unique_texts, row_cluster_labels))
        df[self.cluster_col] = df['expanded_features'].map(cluster_mapping)
        
        print(f"Added '{self.cluster_col}' labels to the DataFrame.")

        # Generate a sanity check
        self._generate_sanity_check(df)
        
        return df, self.cluster_col

    def _generate_sanity_check(self, df):
        """Builds a quick console readout of the row clusters."""
        lines = []
        lines.append(f"\n{'='*30} ROW CLUSTER SAMPLES {'='*30}")
        
        valid_clusters = df[self.cluster_col].dropna().unique()
        n_samples = min(3, len(valid_clusters))
        
        if n_samples == 0:
            self.sanity_check_text = "No clusters found."
            return

        sampled_ids = np.random.choice(valid_clusters, n_samples, replace=False)

        for c_id in sorted(sampled_ids):
            cluster_df = df[df[self.cluster_col] == c_id]
            all_unique_playlists = cluster_df['playlistname'].unique()
            lines.append(f"\n[Row Cluster {int(c_id)}] ({len(all_unique_playlists):,} Unique Contexts)")
            
            # Print a few example playlists from this bicluster
            for playlist in all_unique_playlists[:5]:
                lines.append(f"  🎵 Context: {str(playlist)}")
                
        lines.append(f"{'='*80}")
        
        self.sanity_check_text = "\n".join(lines)
        print(self.sanity_check_text)

    def create_report(self):
        """Generates the isolated text report for Spectral Biclustering."""
        print(f"\nGenerating report for {self.algo_name}...")
        
        report_path = os.path.join(self.report_dir, "run_summary.txt")
        
        with open(report_path, "w") as f:
            f.write(f"Algorithm: {self.algo_name}\n")
            f.write(f"Configuration: {self.config_name}\n")
            f.write(f"Row Clusters: {self.n_row_clusters}\n")
            f.write(f"Column (Word) Clusters: {self.n_column_clusters}\n")
            f.write(f"Random State: {self.random_state}\n")
            f.write("\n")
            f.write(self.sanity_check_text)
            
        print(f"Report saved to {report_path}")