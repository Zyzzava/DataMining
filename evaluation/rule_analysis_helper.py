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

    def plot_metric_distributions(self, output_path='metric_distributions.png'):
        supp_col = self._get_supp_col()
        # Dynamically build the list of columns that actually exist in the data
        potential_metrics = [supp_col, 'confidence', 'lift']
        available_metrics = [m for m in potential_metrics if m and m in self.all_rules.columns]
        
        if not available_metrics:
            print("No metrics found to plot.")
            return

        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 5))
        if len(available_metrics) == 1: axes = [axes]
        
        colors = {'confidence': 'green', 'lift': 'red', 'support': 'blue', 'antecedent support': 'blue'}
        
        for i, col in enumerate(available_metrics):
            sns.histplot(self.all_rules[col], kde=True, ax=axes[i], color=colors.get(col, 'gray'))
            axes[i].set_title(f'Global {col.replace("_", " ").capitalize()} Distribution')
            
        plt.tight_layout()
        plt.savefig(output_path)

    def plot_support_vs_confidence(self, output_path='support_vs_confidence.png'):
        supp_col = self._get_supp_col()
        if not supp_col or 'confidence' not in self.all_rules.columns:
            print("Missing support or confidence columns.")
            return

        plt.figure(figsize=(10, 6))
        # Safely use 'lift' as the color scale if it exists
        c_val = self.all_rules['lift'] if 'lift' in self.all_rules.columns else 'blue'
        
        scatter = plt.scatter(self.all_rules[supp_col], 
                              self.all_rules['confidence'], 
                              c=c_val, 
                              cmap='viridis' if 'lift' in self.all_rules.columns else None, 
                              alpha=0.5, s=10)
        
        if 'lift' in self.all_rules.columns:
            plt.colorbar(scatter, label='Lift')
            
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