import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import Birch
from tqdm import tqdm
from interface.base_algorithm import BaseAlgorithm

class BirchClustering(BaseAlgorithm):
    def __init__(self, k=34, threshold=0.9, branching_factor=25, batch_size=1000):
        config_str = f"k{k}_thresh{threshold}_branch{branching_factor}"

        super().__init__(algo_name="BIRCH", config_name=config_str)
        
        # Hyperparameters
        self.k = k
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.batch_size = batch_size
        
        # State variables to store data for the report
        self.cluster_labels = None
        self.actual_clusters_found = 0
        self.unique_texts_count = 0

    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        self.unique_texts_count = len(unique_texts)
        print(f"\nApplying batched BIRCH clustering... on {self.unique_texts_count:,} unique contexts")
        
        # Instantiate the BIRCH model
        birch = Birch(
            n_clusters=self.k,
            threshold=self.threshold,
            branching_factor=self.branching_factor,
        )

        n_samples = tfidf_matrix.shape[0]

        print("Fitting BIRCH model...")
        for i in tqdm(range(0, n_samples, self.batch_size), desc="Partial Fitting BIRCH"):
            batch = tfidf_matrix[i : i + self.batch_size]
            birch.partial_fit(batch)

        print("Predicting labels...")
        self.cluster_labels = birch.predict(tfidf_matrix)
        self.actual_clusters_found = len(set(self.cluster_labels))
        
        print(f"Assigned cluster labels for {len(self.cluster_labels):,} unique contexts.")
        
        print("Building cluster mapping...")
        cluster_mapping = {
            text: label for text, label in tqdm(zip(unique_texts, self.cluster_labels), 
            total=len(unique_texts), desc="Zipping labels")
        }

        print(f"Broadcasting labels to {len(df):,} rows...")
        tqdm.pandas(desc="Mapping Clusters")
        df[f'birch_cluster_{self.k}'] = df['expanded_features'].progress_map(cluster_mapping.get)    

        print(f"Added 'birch_cluster_{self.k}' labels to the DataFrame.")
        target_col = f'birch_cluster_{self.k}'
        return df, target_col

    def create_report(self):
        """Generates the isolated reports and graphs for BIRCH."""
        print(f"\nGenerating report for {self.algo_name}...")
        
        # 1. Generate Text Summary
        report_path = f"{self.report_dir}/run_summary.txt"
        with open(report_path, "w") as f:
            f.write(f"--- {self.algo_name} Clustering Report ---\n")
            f.write(f"Target Clusters (k)  : {self.k}\n")
            f.write(f"Actual Clusters Found: {self.actual_clusters_found}\n")
            f.write(f"Threshold            : {self.threshold}\n")
            f.write(f"Branching Factor     : {self.branching_factor}\n")
            f.write(f"Batch Size           : {self.batch_size}\n")
            f.write(f"Total Contexts       : {self.unique_texts_count:,}\n")
            
        print(f"Report successfully saved to '{self.report_dir}/'")