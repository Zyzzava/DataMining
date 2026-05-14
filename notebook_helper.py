from evaluation.evaluator import eval
import os
import pickle

def execution_pipeline(algo, df, unique_texts, tfidf_matrix):
    algo_name = algo.algo_name
    col_name = getattr(algo, "cluster_col", None)

    print(f"Checking for: '{col_name}' in columns: {df.columns.tolist()[:10]}...")
    if col_name in df.columns:
        print(f"\n[SKIP] {algo_name} already exists in column '{col_name}'.")
    else:
        print(f"\n{'='*50}")
        print(f"Executing Pipeline for: {algo_name}")
        print(f"{'='*50}")

        df, target_col = algo.run_pipeline(df, unique_texts, tfidf_matrix)
        algo.create_report()

        if target_col and target_col in df.columns:
            from preprocessing.preprocessor import FULLY_PROCESSED_PARQUET
            print(f"[INFO] Saving updated labels to {FULLY_PROCESSED_PARQUET}...")
            df.to_parquet(FULLY_PROCESSED_PARQUET, index=False)

def evaluation_pipeline(algo, df, unique_texts, tfidf_matrix):
    algo_name = algo.algo_name
    col_name = getattr(algo, "cluster_col", None)

    # Define the output paths first
    algo_report_out = algo.report_dir
    os.makedirs(algo_report_out, exist_ok=True)
    eval_report_path = os.path.join(algo_report_out, f"evaluation_metrics_{col_name}.txt")

    # Correct Check: Does the text report already exist on disk?
    if os.path.exists(eval_report_path):
        print(f"\n[SKIP] Evaluation for {algo_name} already exists at '{eval_report_path}'.")
    else:
        # Safety Check: We can't evaluate if the clustering column isn't there
        if col_name not in df.columns:
            print(f"\n[ERROR] Cannot evaluate. Column '{col_name}' is missing from the DataFrame.")
        else:
            # Running evaluation and saving the report
            print(f"\n[INFO] Evaluating {algo_name} results and saving report to '{eval_report_path}'...") 
            eval(df=df,
                cluster_col=col_name,
                unique_texts=unique_texts,
                tfidf_matrix=tfidf_matrix,
                sample_frac=0.1,
                output_dir=algo_report_out)
    return algo_report_out

def build_graph(graph_builder, unique_texts, tfidf_matrix):
    graph_config_name = f"k{graph_builder.k}_sim{graph_builder.sim_threshold}_N{len(unique_texts)}"

    graph_save_dir = os.path.join("graph", "knn", "saved_graphs")
    os.makedirs(graph_save_dir, exist_ok=True)

    graph_save_path = os.path.join(graph_save_dir, f"knn_{graph_config_name}.pkl")

    if os.path.exists(graph_save_path):
        print(f"\n[INFO] Loading previously built k-NN graph from {graph_save_path}...")
        with open(graph_save_path, "rb") as f:
            graph_builder.G = pickle.load(f)
    else:
        print("\n[INFO] Building shared k-NN graph for graph-based clustering...")
        graph_builder.build_graph(tfidf_matrix, unique_texts)
        
        print(f"[INFO] Saving generated k-NN graph to {graph_save_path}...")
        with open(graph_save_path, "wb") as f:
            pickle.dump(graph_builder.G, f)
    
    return graph_config_name

def build_digraph(digraph_builder, unique_texts, tfidf_matrix):
    digraph_config_name = f"k{digraph_builder.k}_sim{digraph_builder.sim_threshold}_N{len(unique_texts)}_mknn"

    digraph_save_dir = os.path.join("graph", "knn", "saved_graphs")
    os.makedirs(digraph_save_dir, exist_ok=True)

    digraph_save_path = os.path.join(digraph_save_dir, f"digraph_{digraph_config_name}.pkl")

    if os.path.exists(digraph_save_path):
        print(f"\n[INFO] Loading previously built directed k-NN graph from {digraph_save_path}...")
        with open(digraph_save_path, "rb") as f:
            digraph_builder.DiG = pickle.load(f)
    else:
        print("\n[INFO] Building shared directed k-NN graph for graph-based clustering...")
        
        # FIX: Explicitly call the directed graph builder function
        digraph_builder.build_directed_graph(tfidf_matrix, unique_texts)
        
        print(f"[INFO] Saving generated directed k-NN graph to {digraph_save_path}...")
        with open(digraph_save_path, "wb") as f:
            pickle.dump(digraph_builder.DiG, f)
    
    return digraph_config_name

import re
import pandas as pd
import numpy as np

def parse_eval_file(filepath):
    """Parses the Top-K list format, accounting for non-breaking spaces and numpy types."""
    data = {}
    # \s+ handles standard spaces, tabs, and non-breaking spaces (\xa0)
    # This looks for "Top-X Average", optional whitespace, a colon, and the bracketed list
    pattern = re.compile(r"(Top-\w+)\s+Average\s*:\s*\[(.*?)\]", re.DOTALL)
    
    with open(filepath, 'r') as f:
        content = f.read()
        matches = pattern.findall(content)
        
        for label, values_str in matches:
            # 1. Remove "np.float64(" and ")"
            clean_str = values_str.replace("np.float64(", "").replace(")", "")
            # 2. Split by comma and convert to float
            try:
                values = [float(v.strip()) for v in clean_str.split(',') if v.strip()]
                data[label] = values
            except ValueError as e:
                print(f"Error parsing values for {label}: {e}")
                
    return data

def compare_results(pure_cf_path, hybrid_path):
    # Load raw data from files
    pure_cf = parse_eval_file(pure_cf_path)
    hybrid = parse_eval_file(hybrid_path)
    
    p_values = [0.1, 0.3, 0.5, 0.7, 1.0] #
    rows = []

    # 1. Build the raw rows
    for category in ['Top-1', 'Top-5', 'Top-10', 'Top-all']:
        cf_vals = pure_cf.get(category, [0]*5)
        hy_vals = hybrid.get(category, [0]*5)
        
        for i, p in enumerate(p_values):
            diff = hy_vals[i] - cf_vals[i]
            rows.append({
                "Metric Depth (p)": p,
                "Category": category,
                "Pure CF": cf_vals[i],
                "Hybrid (Rules+CF)": hy_vals[i],
                "Delta": diff
            })

    # 2. Create DataFrame
    df_compare = pd.DataFrame(rows)
    
    df_compare['Metric Depth (p)'] = df_compare['Metric Depth (p)'].map(lambda x: f"{x:.1f}")
    
    # Pivot for a nicer "Paper-style" view
    pivot_df = df_compare.pivot(
        index="Category", 
        columns="Metric Depth (p)", 
        values=["Delta"]
    )
    
    # 3. Reindex to ensure logical order (Top-1 to Top-all)
    logical_order = ['Top-1', 'Top-5', 'Top-10', 'Top-all']
    pivot_df = pivot_df.reindex(logical_order)

    # 4. Define Formatting Function
    def color_delta(val):
        """Colors positive values green and negative values red."""
        color = '#2ecc71' if val > 0 else '#e74c3c' if val < 0 else 'black'
        return f'color: {color}; font-weight: bold'

    # 5. Apply Styling (Note: mapping the color_delta to Delta columns)
    styled_df = pivot_df.style.format(precision=4) \
        .map(color_delta, subset=['Delta']) \
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ])

    return pivot_df, styled_df

import pandas as pd

def inspect_cluster_playlists(
    df: pd.DataFrame, 
    cluster_col: str, 
    cluster_label: float, 
    playlist_name_col: str, 
    feature_col: str = None, 
    n_samples: int = 5, 
    random_state: int = 42
) -> pd.DataFrame:
    # Filter the DataFrame for the specific cluster
    cluster_entries = df[df[cluster_col] == cluster_label]
    
    # Handle cases where the cluster has fewer entries than requested samples
    n_samples = min(n_samples, len(cluster_entries))
    
    if n_samples == 0:
        print(f"No entries found for cluster {cluster_label} in column '{cluster_col}'.")
        return pd.DataFrame()
        
    # Sample the data to get a diverse representation
    sample_df = cluster_entries.sample(n=n_samples, random_state=random_state)
    
    # Construct a new DataFrame for clean display
    display_data = {
        'Original Index': sample_df.index,
        'Playlist Name': sample_df.get(playlist_name_col, "Unknown Playlist")
    }
    output_df = pd.DataFrame(display_data)
    
    # Optionally format and add the features column
    if feature_col and feature_col in df.columns:
        output_df['Features'] = sample_df[feature_col].apply(
            lambda x: str(x)[:200] + "..." if len(str(x)) > 200 else str(x)
        )
        
    # Drop the default index for a cleaner table look
    return output_df.reset_index(drop=True)