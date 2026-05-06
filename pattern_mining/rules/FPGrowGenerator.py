from pattern_mining.BaseRuleGenerator import BaseRuleGenerator
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pandas as pd
import time
import os
import ast

class FPGrowthGenerator(BaseRuleGenerator):
    def __init__(self, min_support_pct=0.05, min_confidence=0.1, config_name="default", verbose=True):
        # Pass the name up to the base class to handle folder creation
        super().__init__(algo_name="FPGrowth", config_name=config_name)
        
        self.min_support_pct = min_support_pct
        self.min_confidence = min_confidence
        self.verbose = verbose # Toggle for training debug logs

    def mine_rules(self, train_dict, cluster_id, use_cache=True):
        """
        Mines rules or loads them from disk if they already exist.
        """
        start_time = time.time()
        file_path = os.path.join(self.report_dir, f"cluster_{cluster_id}_rules.csv")

        # --- CACHE LOGIC: Read from CSV if it exists ---
        if use_cache and os.path.exists(file_path):
            if self.verbose:
                print(f"[FP-Growth | Cluster {cluster_id}] Found cached rules! Loading from CSV...")
            
            rules_df = pd.read_csv(file_path)

            if not rules_df.empty:
                # Helper function to convert the literal string "frozenset({'X'})" back to a Python frozenset
                def _parse_fs(fs_string):
                    if pd.isna(fs_string): return frozenset()
                    # Strip the word 'frozenset(' and the trailing ')'
                    clean_str = str(fs_string).replace("frozenset(", "").rstrip(")")
                    if clean_str == 'set()': return frozenset()
                    try:
                        # ast.literal_eval safely turns "{'A', 'B'}" into a Python set
                        return frozenset(ast.literal_eval(clean_str))
                    except (ValueError, SyntaxError):
                        return frozenset()

                # Apply the fix to the required columns
                rules_df['antecedents'] = rules_df['antecedents'].apply(_parse_fs)
                rules_df['consequents'] = rules_df['consequents'].apply(_parse_fs)

            self.cluster_rules[cluster_id] = rules_df
            
            if self.verbose:
                print(f"[FP-Growth | Cluster {cluster_id}] Loaded {len(rules_df)} rules from cache in {time.time() - start_time:.2f} seconds.")
            return

        if self.verbose:
            print(f"\n[FP-Growth | Cluster {cluster_id}] --- Starting Rule Mining ---")
            print(f"[FP-Growth | Cluster {cluster_id}] 1. Parsing {len(train_dict)} playlists...")
            
        transactions = [
            [str(track) for track in playlist if pd.notna(track) and track is not None]
            for playlist in train_dict.values()
        ]
        
        if self.verbose:
            print(f"[FP-Growth | Cluster {cluster_id}] 2. One-hot encoding transactions...")
            
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions, sparse=True)
        df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
        
        if self.verbose:
            print(f"[FP-Growth | Cluster {cluster_id}]    -> Matrix shape: {df.shape[0]} playlists x {df.shape[1]} unique tracks.")
            print(f"[FP-Growth | Cluster {cluster_id}] 3. Finding frequent itemsets (min_support={self.min_support_pct})...")
            
        frequent_itemsets = fpgrowth(df, min_support=self.min_support_pct, use_colnames=True, max_len=3)
        
        if self.verbose:
            print(f"[FP-Growth | Cluster {cluster_id}]    -> Found {len(frequent_itemsets)} frequent itemsets.")
            print(f"[FP-Growth | Cluster {cluster_id}] 4. Generating association rules (min_confidence={self.min_confidence})...")
            
        if frequent_itemsets.empty:
            if self.verbose: print(f"[FP-Growth | Cluster {cluster_id}]    -> WARNING: 0 rules generated.")
            self.cluster_rules[cluster_id] = pd.DataFrame()
        else:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
            
            # Optional: Prune weak rules based on lift to keep the cache clean
            if not rules.empty and 'lift' in rules.columns:
                rules = rules[rules['lift'] > 1.2]
                
            self.cluster_rules[cluster_id] = rules
            if self.verbose: 
                print(f"[FP-Growth | Cluster {cluster_id}]    -> Successfully generated {len(rules)} rules.")
            
        self.save_cluster_rules(cluster_id)
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"[FP-Growth | Cluster {cluster_id}] --- Finished in {elapsed:.2f} seconds ---\n")

    def predict(self, seed_tracks, cluster_id, max_recommendations=None, verbose_predict=False):
        """
        verbose_predict: Set to True ONLY if testing a single user. 
        If running a loop of 200 users, keep False to avoid console flooding.
        """
        rules_df = self.cluster_rules.get(cluster_id, pd.DataFrame())
        
        if rules_df.empty:
            if verbose_predict: print(f"[Predict | Cluster {cluster_id}] No rules available. Returning 0 recommendations.")
            return []
            
        recommendations = []
        seed_set = frozenset([str(t) for t in seed_tracks]) # Ensure strings to match rules
        rules_triggered = 0
        
        # Sort by confidence so the best rules are checked first
        sorted_rules = rules_df.sort_values(by='confidence', ascending=False)
        
        for _, rule in sorted_rules.iterrows():
            if rule['antecedents'].issubset(seed_set):
                rules_triggered += 1
                for track in rule['consequents']:
                    if track not in seed_set and track not in recommendations:
                        recommendations.append(track)
                        
                        if max_recommendations and len(recommendations) >= max_recommendations:
                            if verbose_predict: 
                                print(f"[Predict] Hit max_rec limit ({max_recommendations}). Triggered {rules_triggered} rules.")
                            return recommendations
                            
        if verbose_predict:
            print(f"[Predict | Cluster {cluster_id}] Seed tracks: {len(seed_set)}. Triggered {rules_triggered} rules. Found {len(recommendations)} unique recs.")
            
        return recommendations