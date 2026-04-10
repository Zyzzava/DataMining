import os
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    def __init__(self, algo_name, config_name="default"):
        self.algo_name = algo_name
        self.config_name = config_name
        
        self.report_dir = f"clustering/reports/{self.algo_name}/{self.config_name}"
        os.makedirs(self.report_dir, exist_ok=True)

    @abstractmethod
    def run_pipeline(self, df, unique_texts, tfidf_matrix):
        pass

    @abstractmethod
    def create_report(self):
        pass