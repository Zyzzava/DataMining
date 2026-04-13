#!/usr/bin/env python3
"""
Usage:
  python scripts/dataset_stats.py data/spotify_fully_processed.parquet
"""

import sys
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) < 2:
        path = 'data/spotify_fully_processed.parquet'
    else:
        path = sys.argv[1]

    print(f"{'='*50}")
    print(f"LOADING DATASET: {path}")
    print(f"{'='*50}")
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. High Level Dimensions
    total_rows = len(df)
    unique_users = df['user_id'].nunique() if 'user_id' in df.columns else "N/A"
    unique_tracks = df['trackname'].nunique() if 'trackname' in df.columns else "N/A"
    unique_playlists = df['playlistname'].nunique() if 'playlistname' in df.columns else "N/A"

    print("\n--- DATASET OVERVIEW ---")
    print(f"Total Interactions (rows): {total_rows:,}")
    print(f"Unique Users:            {unique_users:,}")
    print(f"Unique Tracks:           {unique_tracks:,}")
    print(f"Unique Raw Playlists:    {unique_playlists:,}")

    # 2. Entity Filtering & Contextual Analysis
    if 'is_contextual' in df.columns:
        contextual_counts = df['is_contextual'].value_counts()
        is_contextual_rows = contextual_counts.get(True, 0)
        
        # Calculate unique playlists that are contextual
        unique_context_pls = df[df['is_contextual'] == True]['playlistname'].nunique()
        
        print("\n--- ENTITY FILTERING (is_contextual) ---")
        print(f"Contextual Interactions: {is_contextual_rows:,} ({is_contextual_rows/total_rows:.1%})")
        print(f"Non-Contextual Rows:     {contextual_counts.get(False, 0):,}")
        print(f"Contextual Playlists:    {unique_context_pls:,} ({unique_context_pls/unique_playlists:.1%})")

    # 3. Preprocessing Feature Stats
    print("\n--- PREPROCESSING COVERAGE ---")
    cols_to_check = ['homogenized_playlist', 'filtered_playlist', 'expanded_features']
    for col in cols_to_check:
        if col in df.columns:
            filled = df[col].notna().sum()
            print(f"{col:25}: {filled:,} rows populated")

    # 4. Top Contextual Examples
    if 'expanded_features' in df.columns:
        print("\n--- TOP 10 EXPANDED CONTEXTS (by interaction count) ---")
        top_contexts = df[df['is_contextual'] == True]['expanded_features'].value_counts().head(10)
        print(top_contexts)

    # 5. Clustering Coverage (if any results exist)
    cluster_cols = [c for c in df.columns if 'cluster' in c or 'community' in c]
    if cluster_cols:
        print("\n--- CLUSTERING STATUS ---")
        for col in cluster_cols:
            assigned = df[col].notna().sum()
            unique_labels = df[col].nunique()
            print(f"{col:30}: {assigned:,} rows assigned to {unique_labels} clusters")

    # 6. Null Value Check
    print("\n--- MISSING VALUES PER COLUMN ---")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()