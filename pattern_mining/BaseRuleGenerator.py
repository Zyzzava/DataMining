import os
import pandas as pd
from abc import ABC, abstractmethod

class BaseRuleGenerator(ABC):
    def __init__(self, algo_name, config_name="default"):
        self.algo_name = algo_name
        self.config_name = config_name
        
        # Keep pattern mining reports separate from clustering reports
        self.report_dir = f"pattern_mining/reports/{self.algo_name}/{self.config_name}"
        os.makedirs(self.report_dir, exist_ok=True)
        
        # A dictionary to store the generated rules for each cluster
        # Format: { cluster_id : pandas.DataFrame(rules) }
        self.cluster_rules = {}

    @abstractmethod
    def mine_rules(self, train_dict, cluster_id):
        """
        Takes the training data for a specific cluster, mines the rules, 
        and stores the resulting DataFrame in self.cluster_rules[cluster_id].
        
        Args:
            train_dict (dict): {playlist_id: [list of track_ids]}
            cluster_id (int/str): The ID of the current cluster.
        """
        pass

    @abstractmethod
    def predict(self, seed_tracks, cluster_id, max_recommendations=None):
        """
        Uses the rules stored for the specific cluster to predict new tracks 
        based on the provided seed tracks.
        
        Args:
            seed_tracks (list/set): The visible tracks in the test playlist.
            cluster_id (int/str): The ID of the cluster this playlist belongs to.
            max_recommendations (int): Optional cap on how many tracks to return.
            
        Returns:
            list: Ranked list of recommended track IDs.
        """
        pass

    def save_cluster_rules(self, cluster_id):
        """
        Utility function to save the generated rules to disk for the report.
        """
        if cluster_id in self.cluster_rules and not self.cluster_rules[cluster_id].empty:
            file_path = os.path.join(self.report_dir, f"cluster_{cluster_id}_rules.csv")
            self.cluster_rules[cluster_id].to_csv(file_path, index=False)
            print(f"Saved rules for cluster {cluster_id} to {file_path}")
        else:
            print(f"No rules to save for cluster {cluster_id}")