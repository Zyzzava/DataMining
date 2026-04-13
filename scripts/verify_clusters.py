#!/usr/bin/env python3
"""
Usage:
  python verify_clusters.py /path/to/data.parquet spectral_cluster_55
  python verify_clusters.py data/spotify_final_healed.parquet spectral_cluster_55
If the file is CSV, it will still work.
"""
import sys
import pandas as pd

def load_df(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if path.endswith('.csv') or path.endswith('.txt'):
        return pd.read_csv(path)
    # try parquet first, then csv
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_clusters.py <data_path> <cluster_col>")
        sys.exit(1)

    data_path = sys.argv[1]
    cluster_col = sys.argv[2]

    print(f"Loading: {data_path}")
    df = load_df(data_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # Top-level checks
    print("Sample rows:")
    print(df.head(5).to_string(index=False), "\n")

    # Required columns check
    for col in ('playlistname', 'trackname', 'user_id', 'expanded_features'):
        print(f"{col:15}: {'FOUND' if col in df.columns else 'MISSING'}")
    print()

    if cluster_col not in df.columns:
        print(f"Cluster column '{cluster_col}' not found in dataframe. Available cluster-like columns:")
        print([c for c in df.columns if 'cluster' in c])
        sys.exit(1)

    # Basic cluster distribution (rows)
    cluster_counts = df[cluster_col].dropna().value_counts().sort_values(ascending=False)
    print("Top 10 clusters by ROW count:")
    print(cluster_counts.head(10).to_string(), "\n")

    # Unique playlists per cluster
    if 'playlistname' in df.columns:
        up = df.groupby(cluster_col)['playlistname'].nunique().sort_values(ascending=False)
        print("Top 10 clusters by UNIQUE playlists:")
        print(up.head(10).to_string(), "\n")
    else:
        print("Column 'playlistname' not present; skipping unique-playlist counts.\n")

    # Unique tracks per cluster
    if 'trackname' in df.columns:
        ut = df.groupby(cluster_col)['trackname'].nunique().sort_values(ascending=False)
        print("Top 10 clusters by UNIQUE tracks:")
        print(ut.head(10).to_string(), "\n")
    else:
        print("Column 'trackname' not present; skipping unique-track counts.\n")

    # Show sample cluster contents for largest cluster
    largest = cluster_counts.index[0]
    print(f"Showing up to 10 sample rows from largest cluster: {largest}")
    print(df[df[cluster_col] == largest].head(10)[['playlistname','trackname','user_id',cluster_col]].to_string(index=False))

if __name__ == '__main__':
    main()