import pandas as pd
from splitter import create_train_test_dicts

def eval(cluster_col='cluster_id'):    
    df = pd.read_parquet('spotify_clustered_master.parquet')

    contextual_df = df[df['is_contextual'] == True]
    unique_clusters = contextual_df[cluster_col].dropna().unique()

    # Use a different variable name for the actual loop!
    for current_cluster_id in unique_clusters:
        print(f"\nProcessing Cluster {current_cluster_id}...")

        cluster_data = contextual_df[contextual_df[cluster_col] == current_cluster_id]
        print(f"Number of samples in Cluster {current_cluster_id}: {len(cluster_data)}")

        # transform and split
        train_dict, test_dict = create_train_test_dicts(cluster_data)