import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class AssociationRuleAnalyzer:
    """Utility to analyze and plot association rules from a consolidated Parquet file."""
    
    def __init__(self, parquet_path="data/mined_rules.parquet"):
        self.parquet_path = parquet_path
        self.all_rules = self._load_data()

    def _load_data(self):
        df = pd.read_parquet(self.parquet_path)
        
        # 1. Clean invisible characters and normalize
        df.columns = [str(c).strip().lower() for c in df.columns]
                
        return df

    def _get_supp_col(self):
        """Standardizes the support column name based on availability."""
        if 'support' in self.all_rules.columns:
            return 'support'
        return 'antecedent support' if 'antecedent support' in self.all_rules.columns else None

    def plot_support_confidence_distributions(self, output_path='evaluation/reports/Hybrid_FPGrowth_CF/support_confidence_distributions.png'):
        """Plots the global distributions for Support and Confidence."""
        supp_col = self._get_supp_col()
        if not supp_col or 'confidence' not in self.all_rules.columns:
            print("Missing support or confidence columns.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metrics = [(supp_col, 'blue'), ('confidence', 'green')]
        
        for i, (col, color) in enumerate(metrics):
            sns.histplot(self.all_rules[col], kde=True, ax=axes[i], color=color)    
            axes[i].set_title(f'Global {col.replace("_", " ").capitalize()} Distribution')
            
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_lift_distribution(self, output_path='evaluation/reports/Hybrid_FPGrowth_CF/lift_distribution.png'):
        """Plots the global distribution specifically for Lift."""
        if 'lift' not in self.all_rules.columns:
            print("Lift column not found.")
            return

        plt.figure(figsize=(7, 5))
        sns.histplot(self.all_rules['lift'], kde=True, color='red')
        plt.title('Global Lift Distribution')
        plt.xlabel('Lift')
        
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_support_vs_confidence(self, output_path='evaluation/reports/Hybrid_FPGrowth_CF/support_vs_confidence.png'):
        supp_col = self._get_supp_col()
        if not supp_col or 'confidence' not in self.all_rules.columns:
            print("Missing support or confidence columns.")
            return

        plt.figure(figsize=(10, 6))
        # Removed the lift color parameter and used a fixed color
        plt.scatter(self.all_rules[supp_col], 
                    self.all_rules['confidence'], 
                    color='purple', alpha=0.5, s=10)
        plt.title('Support vs Confidence')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(output_path)

    def plot_quality_diagnostics(self):
        """Standard diagnostics to evaluate rule strength."""
        if 'lift' not in self.all_rules.columns or 'confidence' not in self.all_rules.columns:
            print("Skipping Quality Diagnostics: Required columns missing.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.all_rules, x='confidence', y='lift', 
                        alpha=0.3, hue=self._get_supp_col(), palette='viridis')
        plt.axvline(x=0.1, color='r', linestyle='--', label='Min Confidence')
        plt.yscale('log')
        plt.title('Rule Strength (Lift) vs. Confidence')
        plt.legend()
        plt.savefig('evaluation/reports/Hybrid_FPGrowth_CF/quality_diagnostics.png')

    def print_average_lift(self):
        if 'lift' in self.all_rules.columns:
            avg_lift = self.all_rules['lift'].mean()
            print(f"Average Lift across all clusters: {avg_lift:.4f}")
        else:
            print("Lift column not found in data.")

    def plot_rule_contribution_histogram(self, stats_json_path="evaluation/reports/rule_activation_stats.json", output_path='evaluation/reports/Hybrid_FPGrowth_CF/ rule_contribution_histogram.png'):
        """
        Plots a histogram of the rule_vs_cf_ratio across all clusters and prints the average.
        """
        if not os.path.exists(stats_json_path):
            print(f"Stats file not found: {stats_json_path}")
            return

        with open(stats_json_path, "r") as f:
            import json
            stats = json.load(f)

        # Convert the dictionary to a DataFrame
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        
        ratio_col = 'rule_vs_cf_ratio'
        if ratio_col not in stats_df.columns:
            print(f"Column {ratio_col} not found in stats.")
            return

        # 1. Print the average ratio to the console
        mean_ratio = stats_df[ratio_col].mean()
        print(f"\n{'='*30}\nAVERAGE RULE CONTRIBUTION RATIO: {mean_ratio:.4f}\n{'='*30}")

        # 2. Create the Histogram
        plt.figure(figsize=(10, 6))
        
        # Using pure plt.hist or sns.histplot without the KDE line for a clean histogram
        plt.hist(stats_df[ratio_col], bins=20, color='orange', edgecolor='black', alpha=0.7)
        
        # Add a vertical line for the mean
        plt.axvline(mean_ratio, color='red', linestyle='--', label=f'Mean: {mean_ratio:.4f}')
        
        plt.title('Cluster Distribution: Association Rule Contribution')
        plt.xlabel('Contribution Ratio (Rules / Total Recs)')
        plt.ylabel('Frequency (Number of Clusters)')
        plt.legend()
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Histogram saved to {output_path}")
    
    def print_average_rule_contribution(self, stats_json_path="evaluation/reports/Hybrid_FPGrowth_CF/rule_activation_stats.json"):
        """
        Calculates and prints the average rule_vs_cf_ratio across all clusters.
        """
        import os
        import json
        import pandas as pd

        if not os.path.exists(stats_json_path):
            print(f"Stats file not found: {stats_json_path}")
            return

        with open(stats_json_path, "r") as f:
            stats = json.load(f)

        # Convert the dictionary to a DataFrame
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        
        ratio_col = 'rule_vs_cf_ratio'
        if ratio_col not in stats_df.columns:
            print(f"Column {ratio_col} not found in stats.")
            return

        # Print the average ratio to the console
        mean_ratio = stats_df[ratio_col].mean()
        print(f"\n{'='*30}\nAVERAGE RULE CONTRIBUTION RATIO: {mean_ratio:.4f}\n{'='*30}")

def main():
    analyzer = AssociationRuleAnalyzer()
    analyzer.plot_metric_distributions()
    analyzer.plot_support_vs_confidence()
    analyzer.plot_quality_diagnostics()
    analyzer.print_average_lift()

if __name__ == "__main__":
    main()