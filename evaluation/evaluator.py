import json
import os

import pandas as pd
import numpy as np
from evaluation.metrics import evaluate_metrics
from evaluation.splitter import create_train_test_dict
from evaluation.recommender import get_recommendations
from collections import defaultdict
from tqdm import tqdm
from evaluation.plot_comparison import plot_cluster_distribution, plot_f01_comparison
from evaluation.silhouette import evaluate_silhouette

def eval(df, cluster_col, unique_texts, tfidf_matrix, sample_frac=0.1, output_dir="evaluation/reports", rule_generator=None, refine_results=False):    
    """
    Evaluates the clustering performance and saves results to the algorithm's specific folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving evaluation results to: {output_dir}/")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Sillhouette score evaluation
    print(f"\n{'='*30}\nCALCULATING SILHOUETTE SCORE\n{'='*30}")
    evaluate_silhouette(df, unique_texts=unique_texts, tfidf_matrix=tfidf_matrix, cluster_col=cluster_col)
    ################################

    contextual_df = df[df['is_contextual'] == True]
    # unique clusters sorted largest -> smallest
    unique_clusters = contextual_df[cluster_col].dropna().value_counts().index.tolist()    
    
    # p values are the number of top recommendations to consider when evaluating precision, recall, and f-score
    p_values = [0.1, 0.3, 0.5, 0.7, 1.0]

    cluster_performances = {}
    # Loop through each unique cluster and process the data
    for current_cluster_id in unique_clusters:
        print(f"\nProcessing Cluster {current_cluster_id}...")

        cluster_data = contextual_df[contextual_df[cluster_col] == current_cluster_id]
        print(f"Number of samples in Cluster {current_cluster_id}: {len(cluster_data)}")

        # transform and split
        train_dict, test_dict = create_train_test_dict(cluster_data)
        
        # sub-sample the test users with floor ceiling subsampling
        all_test_users = list(test_dict.keys())
        total_cluster_users = len(all_test_users)
        MIN_USERS_FOR_CF = 50
        if total_cluster_users < MIN_USERS_FOR_CF:
            print(f"  -> Skipping cluster (Only {total_cluster_users} users. CF requires a larger crowd).")
            continue
        TARGET_SAMPLE_SIZE = 200 
        if total_cluster_users > TARGET_SAMPLE_SIZE:
            target_users = np.random.choice(all_test_users, TARGET_SAMPLE_SIZE, replace=False)
            print(f"  -> Sub-sampling to {TARGET_SAMPLE_SIZE} users (out of {total_cluster_users})...")
        else:
            target_users = all_test_users
            print(f"  -> Evaluating all {total_cluster_users} users in this cluster...")

        #build inverted user track index
        track_to_users_index = defaultdict(set)
        for user, tracks in train_dict.items():
            for track in tracks:
                track_to_users_index[track].add(user)

        ### Rule gen PART 3 ### 
        if rule_generator is not None:
            print(f"  -> Mining association rules for cluster {current_cluster_id}...")
            rule_generator.mine_rules(train_dict, current_cluster_id)

        current_cluster_scores = {p: [] for p in p_values} 
        print(f"  -> Generating recommendations & evaluating {len(target_users)} sampled users...")        
        # run recommendations and evaluations for each user in the test set
        for target_user in tqdm(target_users, desc="Evaluating users", leave=False):
            # get standard CF recommendations
            cf_predictions = get_recommendations(target_user, train_dict, track_to_users_index)

            ### Rule gen PART 3 ### 
            if rule_generator is not None:
                seed_tracks = train_dict[target_user] 

                if not refine_results:
                    rule_predictions = rule_generator.predict(seed_tracks, current_cluster_id)
                
                else: 
                    rule_metadata = rule_generator.predict_with_metadata(seed_tracks, current_cluster_id)
                    for p in p_values:
                        # 1. Identify unique items in the cluster using your specific columns
                        # We use trackname as the item identifier here
                        unique_items_in_cluster = cluster_data['trackname'].unique()
                        
                        # 2. Calculate the total recommendation depth for this p-value
                        total_slots = max(1, int(len(unique_items_in_cluster) * p))
                        
                        # 3. Dictate the 'Expert' budget (e.g., max 20% of the slots can be rules)
                        # This prevents the 'Substitution Effect' where rules displace CF results
                        dynamic_max_rules = max(1, int(total_slots * 0.20)) 

                        if rule_generator is not None and refine_results:
                            # 4. Use metadata to filter for 'Bridge' tracks using Lift
                            rule_metadata = rule_generator.predict_with_metadata(seed_tracks, current_cluster_id)
                            
                            # Filter: 1.5 < Lift < 15 to avoid noise and 'Album Effects'
                            best_rules = [
                                r for r in rule_metadata 
                                if 1.5 < r['lift'] < 15.0 
                            ]
                            
                            # 5. Sort by Confidence and apply the dynamic cap
                            best_rules = sorted(best_rules, key=lambda x: x['confidence'], reverse=True)[:dynamic_max_rules]
                            rule_predictions = [r['track'] for r in best_rules]
                        else:
                            # Backwards compatibility for naive concatenation
                            rule_predictions = rule_generator.predict(seed_tracks, current_cluster_id) if rule_generator else []
                
                ranked_predictions = []
                seen_tracks = set(seed_tracks)

                for track in rule_predictions + cf_predictions:
                    if track not in seen_tracks:
                        ranked_predictions.append(track)
                        seen_tracks.add(track) # Add it to the set so we never add it twice!
            else:
                # backwards comp
                ranked_predictions = cf_predictions
            
            #for each p, evaluate the metrics and store metrics for this user
            #metrics include precision, recall, and f-score at the specified p-value
            for p in p_values:
                metrics = evaluate_metrics(
                    ranked_predictions, 
                    test_dict[target_user], 
                    p=p
                )
                
                # Store the score for this specific user at this specific p-value
                current_cluster_scores[p].append(metrics['f_0.1'])

        # After processing all users in the current cluster, calculate the average score for each p-value and store it
        cluster_averages = {p: np.mean(current_cluster_scores[p]) if current_cluster_scores[p] else 0.0 for p in p_values}
        cluster_performances[current_cluster_id] = cluster_averages
    
    # rank clusters based on the average f-score across all p-values
    ranked_clusters = sorted(
        cluster_performances.keys(), 
        key=lambda c_id: np.mean(list(cluster_performances[c_id].values())), 
        reverse=True
    )

    #helper function to graph the average of the top K clusters
    def get_top_k_averages(k):
        top_k_ids = ranked_clusters[:k]
        k_averages = []
        for p in p_values:
            # Get the score at this p-value for the top K clusters, and average them
            avg_at_p = np.mean([cluster_performances[c_id][p] for c_id in top_k_ids])
            k_averages.append(avg_at_p)
        return k_averages

    #get results matching the paper
    results = {
        'top_1': get_top_k_averages(1),
        'top_5': get_top_k_averages(5),
        'top_10': get_top_k_averages(10),
        'top_all': get_top_k_averages(len(ranked_clusters))
    }

    # SAVING THE REPORT
    report_path = os.path.join(output_dir, f"evaluation_metrics_{cluster_col}.txt")
    print(f"\nSaving metrics to {report_path}...")
    
    with open(report_path, "w") as f:
        f.write(f"=== Evaluation Results for {cluster_col} ===\n")
        f.write(f"Total Clusters Evaluated: {len(ranked_clusters)}\n\n")
        f.write(f"Top-1 Average  : {[round(x, 4) for x in results['top_1']]}\n")
        f.write(f"Top-5 Average  : {[round(x, 4) for x in results['top_5']]}\n")
        f.write(f"Top-10 Average : {[round(x, 4) for x in results['top_10']]}\n")
        f.write(f"Top-all Average: {[round(x, 4) for x in results['top_all']]}\n")

    # Save the raw cluster performances to JSON for deeper analysis later
    with open(os.path.join(output_dir, "raw_cluster_scores.json"), "w") as f:
        json.dump(cluster_performances, f, indent=4)

    #Saving the graphs 
    print("\nGenerating evaluation graphs...")
    plot_f01_comparison(
        p_values=p_values, 
        kmeans_scores=results,
        title_prefix=f"Evaluation Results for '{cluster_col}'",
        save_path=os.path.join(output_dir, "f01_comparison.png")
    )

    plot_cluster_distribution(
        df=contextual_df,
        cluster_col=cluster_col,
        save_path=os.path.join(output_dir, "cluster_distribution.png")
    )

    print("\n=== FINAL EVALUATION AVERAGES ===")
    print("Top-1 Average:  ", [round(x, 4) for x in results['top_1']])
    print("Top-5 Average:  ", [round(x, 4) for x in results['top_5']])
    print("Top-all Average:", [round(x, 4) for x in results['top_all']])



