import pandas as pd
import numpy as np
from evaluation.metrics import evaluate_metrics
from evaluation.splitter import create_train_test_dict
from evaluation.recommender import get_recommendations
from collections import defaultdict
from tqdm import tqdm

def eval(cluster_col, sample_frac=0.1):    
    # SEED NP RANDOM
    np.random.seed(42)

    df = pd.read_parquet('data/spotify_fully_processed.parquet')

    contextual_df = df[df['is_contextual'] == True]

    # unique clusters sorted largest -> smallest
    unique_clusters = contextual_df[cluster_col].dropna().value_counts().index.tolist()

    # Loop through each unique cluster and process the data
    for current_cluster_id in unique_clusters:
        print(f"\nProcessing Cluster {current_cluster_id}...")

        cluster_data = contextual_df[contextual_df[cluster_col] == current_cluster_id]
        print(f"Number of samples in Cluster {current_cluster_id}: {len(cluster_data)}")

        # transform and split
        train_dict, test_dict = create_train_test_dict(cluster_data)
        
        # sub-sample the test users
        all_test_users = list(test_dict.keys())
        if sample_frac < 1.0:
            n_samples = int(len(all_test_users) * sample_frac)
            target_users = np.random.choice(all_test_users, n_samples, replace=False)
            print(f"  -> Sub-sampling {sample_frac*100}%: evaluating {len(target_users)} users...")
        else: 
            target_users = all_test_users

        #build inverted user track index
        track_to_users_index = defaultdict(set)
        for user, tracks in train_dict.items():
            for track in tracks:
                track_to_users_index[track].add(user)

        p_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        cluster_scores = {p: [] for p in p_values}
        print(f"  -> Generating recommendations & evaluating {len(target_users)} sampled users...")        
        # run recommendations and evaluations for each user in the test set
        for target_user in tqdm(target_users, desc="Evaluating users", leave=False):
            ranked_predictions = get_recommendations(target_user, train_dict, track_to_users_index)
            
            #for each p, evaluate the metrics and store metrics for this user
            #metrics include precision, recall, and f-score at the specified p-value
            for p in p_values:
                metrics = evaluate_metrics(
                    ranked_predictions, 
                    test_dict[target_user], 
                    p=p
                )
                
                # Store the score for this specific user at this specific p-value
                cluster_scores[p].append(metrics)
        
        # print average scores for this cluster at each p-value
        for p in p_values:
            avg_f_score = np.mean([user_score['f_0.1'] for user_score in cluster_scores[p]])            
            print(f"    - p={p:<3} | Avg F0.1 Score: {avg_f_score:.4f}")


