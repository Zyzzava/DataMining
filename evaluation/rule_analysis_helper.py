import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AssociationRuleAnalyzer:
    """Utility to analyze and plot association rules from multiple cluster files."""
    
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.all_rules = self._load_data()

    def _load_data(self):
        all_dfs = []
        files = [f for f in os.listdir(self.directory_path) 
                 if f.startswith('cluster_') and f.endswith('_rules.csv')]
        
        for file in files:
            # Extract cluster ID from filename like 'cluster_20.0_rules.csv'
            cluster_id = file.split('_')[1]
            df = pd.read_csv(os.path.join(self.directory_path, file))
            df['cluster'] = cluster_id
            all_dfs.append(df)
            
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def get_summary_statistics(self):
        if self.all_rules.empty: return "No data found."
        return self.all_rules.describe()

    def plot_rules_per_cluster(self, output_path='rules_per_cluster.png'):
        counts = self.all_rules['cluster'].value_counts().sort_index()
        plt.figure(figsize=(14, 6))
        counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Recommendation Density: Rules per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Count of Generated Rules')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_metric_distributions(self, output_path='metric_distributions.png'):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = [('support', 'blue'), ('confidence', 'green'), ('lift', 'red')]
        
        for i, (col, color) in enumerate(metrics):
            sns.histplot(self.all_rules[col], kde=True, ax=axes[i], color=color)
            axes[i].set_title(f'Global {col.capitalize()} Distribution')
            
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_support_vs_confidence(self, output_path='support_vs_confidence.png'):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.all_rules['support'], 
                              self.all_rules['confidence'], 
                              c=self.all_rules['lift'], 
                              cmap='viridis', alpha=0.5, s=10)
        plt.colorbar(scatter, label='Lift')
        plt.title('Support vs Confidence (Colored by Lift)')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(output_path)
    
    def plot_normalized_density(self):
        """Visualizes which clusters are 'predictable' vs 'stochastic'."""
        counts = self.all_rules['cluster'].astype(float).astype(int).value_counts().sort_index()
        plt.figure(figsize=(14, 6))
        sns.barplot(x=counts.index, y=counts.values, palette="magma")
        plt.title('Predictive Density: Rules per Cluster\n(Threshold 1.5% scales with size)')
        plt.ylabel('Number of Significant Rules')
        plt.savefig('predictive_density.png')

    def plot_quality_diagnostics(self):
        """Evaluates if the 30% Confidence threshold is actually finding strong patterns."""
        plt.figure(figsize=(10, 6))
        # Lift > 1 means the association is stronger than random chance
        plt.scatter(self.all_rules['confidence'], self.all_rules['lift'], 
                    alpha=0.3, c=self.all_rules['support'], cmap='viridis')
        plt.axvline(x=0.3, color='r', linestyle='--', label='Min Confidence')
        plt.yscale('log') # Lift can have high variance
        plt.title('Rule Strength (Lift) vs. Confidence')
        plt.xlabel('Confidence (Likelihood)')
        plt.ylabel('Lift (Strength relative to size)')
        plt.savefig('quality_diagnostics.png')

    def print_average_lift(self):
        avg_lift = self.all_rules['lift'].mean()
        print(f"Average Lift across all clusters: {avg_lift:.4f}")