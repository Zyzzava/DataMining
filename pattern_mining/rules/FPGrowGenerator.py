from pattern_mining.BaseRuleGenerator import BaseRuleGenerator
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd

class FPGrowthGenerator(BaseRuleGenerator):
    def __init__(self, min_support_pct=0.05, min_confidence=0.1, config_name="default"):
        # Pass the name up to the base class to handle folder creation
        super().__init__(algo_name="FPGrowth", config_name=config_name)
        
        self.min_support_pct = min_support_pct
        self.min_confidence = min_confidence

    def mine_rules(self, train_dict, cluster_id):
        transactions = [
            [str(track) for track in playlist if pd.notna(track) and track is not None]
            for playlist in train_dict.values()
        ]
        
        # Convert to one-hot encoded format required by mlxtend
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Mine patterns
        frequent_itemsets = fpgrowth(df, 
                                     min_support=self.min_support_pct, 
                                     use_colnames=True,
                                     max_len=4)
        
        # Generate rules and store them in the parent class dictionary
        if frequent_itemsets.empty:
            self.cluster_rules[cluster_id] = pd.DataFrame()
        else:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
            self.cluster_rules[cluster_id] = rules
            
        # Automatically save them to the report directory
        self.save_cluster_rules(cluster_id)

    def predict(self, seed_tracks, cluster_id, max_recommendations=None):
        rules_df = self.cluster_rules.get(cluster_id, pd.DataFrame())
        if rules_df.empty:
            return []
            
        recommendations = []
        seed_set = frozenset(seed_tracks)
        
        # Sort by confidence so the best rules are checked first
        sorted_rules = rules_df.sort_values(by='confidence', ascending=False)
        
        for _, rule in sorted_rules.iterrows():
            if rule['antecedents'].issubset(seed_set):
                for track in rule['consequents']:
                    if track not in seed_set and track not in recommendations:
                        recommendations.append(track)
                        if max_recommendations and len(recommendations) >= max_recommendations:
                            return recommendations
                            
        return recommendations