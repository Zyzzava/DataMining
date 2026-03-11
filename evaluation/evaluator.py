import pandas as pd
import numpy as np
from evaluation.metrics import evaluate_metrics
from evaluation.splitter import create_train_test_dict
from evaluation.recommender import get_recommendations

def eval(cluster_col):    
    df = pd.read_parquet('data/spotify_fully_processed.parquet')

    contextual_df = df[df['is_contextual'] == True]
    unique_clusters = contextual_df[cluster_col].dropna().unique()

    # Loop through each unique cluster and process the data
    for current_cluster_id in unique_clusters:
        print(f"\nProcessing Cluster {current_cluster_id}...")

        cluster_data = contextual_df[contextual_df[cluster_col] == current_cluster_id]
        print(f"Number of samples in Cluster {current_cluster_id}: {len(cluster_data)}")

        # transform and split
        train_dict, test_dict = create_train_test_dict(cluster_data)


        p_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        precision_scores = {p: [] for p in p_values}
        recall_scores = {p: [] for p in p_values}
        cluster_f_scores = {p: [] for p in p_values}
        print(f"  -> Generating recommendations & evaluating {len(test_dict)} users...")
        
        # run recommendations and evaluations for each user in the test set
        for target_user in test_dict.keys():
            ranked_predictions = get_recommendations(target_user, train_dict)
            
            #for each p, evaluate the metrics and store metrics for this user
            #metrics include precision, recall, and f-score at the specified p-value
            for p in p_values:
                metrics = evaluate_metrics(
                    ranked_predictions, 
                    test_dict[target_user], 
                    p=p
                )
                
                # Store the score for this specific user at this specific p-value
                precision_scores[p].append(metrics['precision'])
                recall_scores[p].append(metrics['recall'])
                cluster_f_scores[p].append(metrics['f_0.1'])
        
        # print 
        for p in p_values:
            avg_f_score = np.mean(cluster_f_scores[p])
            print(f"    - p={p:<3} | Avg F0.1 Score: {avg_f_score:.4f}")


