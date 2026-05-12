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
        print(df.columns)
                
        return df

    def _get_supp_col(self):
        """Standardizes the support column name based on availability."""
        if 'support' in self.all_rules.columns:
            return 'support'
        return 'antecedent support' if 'antecedent support' in self.all_rules.columns else None

    def plot_support_confidence_distributions(self, output_path='support_confidence_distributions.png'):
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

    def plot_lift_distribution(self, output_path='lift_distribution.png'):
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

    def plot_support_vs_confidence(self, output_path='support_vs_confidence.png'):
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
        plt.savefig('quality_diagnostics.png')

    def print_average_lift(self):
        if 'lift' in self.all_rules.columns:
            avg_lift = self.all_rules['lift'].mean()
            print(f"Average Lift across all clusters: {avg_lift:.4f}")
        else:
            print("Lift column not found in data.")

def main():
    print("herro")
    analyzer = AssociationRuleAnalyzer()
    analyzer.plot_metric_distributions()
    analyzer.plot_support_vs_confidence()
    analyzer.plot_quality_diagnostics()
    analyzer.print_average_lift()

if __name__ == "__main__":
    main()