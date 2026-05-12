from pattern_mining.BaseRuleGenerator import BaseRuleGenerator
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
import time
import os

class FPGrowthGenerator(BaseRuleGenerator):
    def __init__(self, min_support_pct=0.05, min_confidence=0.1, config_name="default", verbose=True):
        super().__init__(algo_name="FPGrowth", config_name=config_name)
        self.min_support_pct = min_support_pct
        self.min_confidence = min_confidence
        self.verbose = verbose
        # Path to the consolidated rule store
        self.parquet_path = "data/mined_rules.parquet"

    def mine_rules(self, train_dict, cluster_id, use_cache=True):
        """
        Loads rules from the master Parquet file or mines them if missing, 
        filtering strictly by support and confidence.
        """
        start_time = time.time()
        cluster_id_str = str(float(cluster_id))

        # --- PARQUET CACHE LOGIC ---
        if use_cache and os.path.exists(self.parquet_path):
            if self.verbose:
                print(f"[FP-Growth | Cluster {cluster_id}] Checking master Parquet...")
            
            full_rules_df = pd.read_parquet(self.parquet_path)
            rules_df = full_rules_df[full_rules_df['cluster_id'] == cluster_id_str].copy()

            if not rules_df.empty:
                # Convert list format from Parquet back to frozensets for prediction logic
                rules_df['antecedents'] = rules_df['antecedents'].apply(frozenset)
                rules_df['consequents'] = rules_df['consequents'].apply(frozenset)
                
                self.cluster_rules[cluster_id] = rules_df
                if self.verbose:
                    print(f"[FP-Growth] Loaded {len(rules_df)} rules from Parquet in {time.time() - start_time:.2f}s.")
                return

        # --- MINING LOGIC ---
        if self.verbose:
            print(f"[FP-Growth | Cluster {cluster_id}] Mining new rules...")

        transactions = [[str(t) for t in p if pd.notna(t)] for p in train_dict.values()]
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions, sparse=True)
        df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

        # 1. Frequent Itemsets based on min_support
        frequent_itemsets = fpgrowth(df, min_support=self.min_support_pct, use_colnames=True, max_len=3)
        
        if frequent_itemsets.empty:
            self.cluster_rules[cluster_id] = pd.DataFrame()
        else:
            # 2. Association Rules based on min_confidence
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
            
            # Lift pruning removed as requested; keeping all rules satisfying support/confidence
            rules['cluster_id'] = cluster_id_str
            self.cluster_rules[cluster_id] = rules

            if self.verbose:
                print(f"[FP-Growth] Generated {len(rules)} rules based on support and confidence.")

    def predict(self, seed_tracks, cluster_id, max_recommendations=None):
        rules_df = self.cluster_rules.get(cluster_id, pd.DataFrame())
        if rules_df.empty:
            return []
            
        recommendations = []
        seed_set = frozenset([str(t) for t in seed_tracks])
        
        # Sort by confidence to prioritize the most likely transitions
        sorted_rules = rules_df.sort_values(by='confidence', ascending=False)
        
        for _, rule in sorted_rules.iterrows():
            if rule['antecedents'].issubset(seed_set):
                for track in rule['consequents']:
                    if track not in seed_set and track not in recommendations:
                        recommendations.append(track)
                        if max_recommendations and len(recommendations) >= max_recommendations:
                            return recommendations
        return recommendations
    
    def predict_with_metadata(self, seed_tracks, cluster_id):
        """
        Extracts rules with metrics. Support is included to track activation statistics.
        """
        rules_df = self.cluster_rules.get(cluster_id, pd.DataFrame())
        if rules_df.empty:
            return []
            
        recommendations_metadata = []
        seed_set = frozenset([str(t) for t in seed_tracks])
        
        # mlxtend uses 'support' for rule-level support (antecedent AND consequent)
        support_col = 'support' if 'support' in rules_df.columns else 'antecedent support'
        
        for _, rule in rules_df.iterrows():
            if rule['antecedents'].issubset(seed_set):
                for track in rule['consequents']:
                    if track not in seed_set:
                        existing_entry = next((item for item in recommendations_metadata if item["track"] == track), None)
                        
                        entry = {
                            'track': track,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule[support_col]
                        }
                        
                        if not existing_entry:
                            recommendations_metadata.append(entry)
                        elif entry['confidence'] > existing_entry['confidence']:
                            existing_entry.update(entry)
                                
        return recommendations_metadata