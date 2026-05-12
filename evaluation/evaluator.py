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
from evaluation.part3_helper import init_trace_tracker, update_trace, log_trace_results

def eval(df, cluster_col, unique_texts, tfidf_matrix, output_dir="evaluation/reports", rule_generator=None, refine_results=False):    
    """
    Evaluates the clustering performance and saves results to the algorithm's specific folder.
    """
    rule_activation_stats = {}

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # Sillhouette score evaluation
    print(f"\n{'='*30}\nCALCULATING SILHOUETTE SCORE\n{'='*30}")
    # saviing silhouette scores to a text file for reference
    silhouette_report_path = os.path.join(output_dir, f"silhouette_scores_{cluster_col}.txt")
    silhouette_score = evaluate_silhouette(df, unique_texts=unique_texts, tfidf_matrix=tfidf_matrix, cluster_col=cluster_col)
    with open(silhouette_report_path, "w") as f:
        f.write(f"Silhouette Score Evaluation for {cluster_col}\n")
        f.write(f"{'-'*50}\n")
        f.write(f"Silhouette Score: {silhouette_score}\n")
    
    contextual_df = df[df['is_contextual'] == True]
    unique_clusters = contextual_df[cluster_col].dropna().value_counts().index.tolist()    
    p_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    cluster_performances = {}

    # Loop through each unique cluster and process the data
    for current_cluster_id in unique_clusters:
        print(f"\nProcessing Cluster {current_cluster_id}...")
        cluster_data = contextual_df[contextual_df[cluster_col] == current_cluster_id]
        print(f"Number of samples in Cluster {current_cluster_id}: {len(cluster_data)}")
        train_dict, test_dict = create_train_test_dict(cluster_data)
        trace_tracker = init_trace_tracker()
        
        all_test_users = list(test_dict.keys())
        total_cluster_users = len(all_test_users)

        MIN_USERS_FOR_CF = 50
        if total_cluster_users < MIN_USERS_FOR_CF: continue
        TARGET_SAMPLE_SIZE = 200 
        target_users = np.random.choice(all_test_users, min(total_cluster_users, TARGET_SAMPLE_SIZE), replace=False)

        #build inverted user track index
        track_to_users_index = defaultdict(set)
        for user, tracks in train_dict.items():
            for track in tracks:
                track_to_users_index[track].add(user)

        # Mine rules 
        if rule_generator is not None:
            rule_generator.mine_rules(train_dict, current_cluster_id)

        # Cluster Rule Stats
        cluster_rule_data = {
            "total_rules_mined": 0,
            "total_recommendations": 0,
            "rule_contributions": 0,
            "lifts": [],
            "confidences": [],
            "supports": []
        }

        current_cluster_scores = {p: [] for p in p_values} 

        print(f"  -> Generating recommendations & evaluating {len(target_users)} sampled users...")        
        # run recommendations and evaluations for each user in the test set
        for target_user in tqdm(target_users, desc="Evaluating users", leave=False):
            # get standard CF recommendations
            cf_predictions = get_recommendations(target_user, train_dict, track_to_users_index)
            seed_tracks = train_dict[target_user] 

            ### Rule gen PART 3 ### 
            rule_predictions = []
            if rule_generator is not None:
                rule_metadata = rule_generator.predict_with_metadata(seed_tracks, current_cluster_id)

                # Filter if refining
                if refine_results:
                    unique_items = cluster_data['trackname'].unique()
                    dynamic_max = max(1, int(len(unique_items) * 1.0 * 0.20)) # 20 %   
                    active_rules = [r for r in rule_metadata if r['lift'] > 2.0]
                    active_rules = sorted(active_rules, key=lambda x: x['confidence'], reverse=True)[:dynamic_max]
                else:
                    active_rules = rule_metadata
                
                rule_predictions = [r['track'] for r in active_rules]

                # Collect stats for JSON
                for r in active_rules:
                    cluster_rule_data["lifts"].append(r['lift'])
                    cluster_rule_data["confidences"].append(r['confidence'])
                    cluster_rule_data["supports"].append(r['support'])

            ranked_predictions = []
            seen_tracks = set(seed_tracks)

            # Tracking ratios
            for track in rule_predictions + cf_predictions:
                if track not in seen_tracks:
                    ranked_predictions.append(track)
                    seen_tracks.add(track)
                    if track in rule_predictions:
                        cluster_rule_data["rule_contributions"] += 1
                    cluster_rule_data["total_recommendations"] += 1
                
            
            update_trace(trace_tracker, ranked_predictions, rule_predictions)
            for p in p_values:
                metrics = evaluate_metrics(ranked_predictions, test_dict[target_user], p=p)
                current_cluster_scores[p].append(metrics['f_0.1'])

        # --- Process Statistics for this Cluster ---
        if cluster_rule_data["total_recommendations"] > 0:
            rule_ratio = cluster_rule_data["rule_contributions"] / cluster_rule_data["total_recommendations"]
            avg_lift = np.mean(cluster_rule_data["lifts"]) if cluster_rule_data["lifts"] else 0
            avg_conf = np.mean(cluster_rule_data["confidences"]) if cluster_rule_data["confidences"] else 0
            avg_supp = np.mean(cluster_rule_data["supports"]) if cluster_rule_data["supports"] else 0
            
            rule_activation_stats[str(current_cluster_id)] = {
                "rule_vs_cf_ratio": round(rule_ratio, 4),
                "avg_lift": round(float(avg_lift), 4),
                "avg_confidence": round(float(avg_conf), 4),
                "avg_support": round(float(avg_supp), 4),
                "total_recs": cluster_rule_data["total_recommendations"]
            }

        log_trace_results(trace_tracker, current_cluster_id)
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

    # --- Save JSON Statistics ---
    stats_path = os.path.join(output_dir, "rule_activation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(rule_activation_stats, f, indent=4)
    print(f"Rule activation stats saved to {stats_path}")
    
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