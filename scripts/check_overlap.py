import pandas as pd

print("Loading datasets (skipping malformed lines)...")

# 1. Load the datasets with 'on_bad_lines' to bypass parsing errors
spotify_df = pd.read_csv('spotify_dataset.csv', skipinitialspace=True, on_bad_lines='skip')
combined_df = pd.read_csv('combined_dataset.csv', skipinitialspace=True, on_bad_lines='skip')

# 2. Rename columns in the Spotify dataset to match the Combined dataset
# First, let's strip any hidden quotes/spaces from column names just in case
spotify_df.columns = spotify_df.columns.str.strip().str.replace('"', '')
combined_df.columns = combined_df.columns.str.strip().str.replace('"', '')

spotify_df.rename(columns={'artistname': 'artist_name', 'trackname': 'track_name'}, inplace=True)

# 3. Clean the text columns to ensure accurate matching 
# (Converting to lowercase and stripping leading/trailing whitespace)
for df in [spotify_df, combined_df]:
    if 'artist_name' in df.columns and 'track_name' in df.columns:
        df['artist_name'] = df['artist_name'].astype(str).str.lower().str.strip()
        df['track_name'] = df['track_name'].astype(str).str.lower().str.strip()
    else:
        print(f"Warning: Columns missing in one of the dataframes. Available columns: {df.columns.tolist()}")

# 4. Perform an inner merge to find the overlap
overlap_df = pd.merge(spotify_df, combined_df, on=['artist_name', 'track_name'], how='inner')

# 5. Calculate unique track overlaps
unique_spotify = spotify_df[['artist_name', 'track_name']].drop_duplicates()
unique_combined = combined_df[['artist_name', 'track_name']].drop_duplicates()
unique_overlap = overlap_df[['artist_name', 'track_name']].drop_duplicates()

# 6. Print the results
print("\n--- OVERLAP ANALYSIS ---")
print(f"Total rows in spotify_dataset: {len(spotify_df):,}")
print(f"Total rows in combined_dataset: {len(combined_df):,}\n")

print(f"Unique tracks in spotify_dataset: {len(unique_spotify):,}")
print(f"Unique tracks in combined_dataset: {len(unique_combined):,}\n")

print(f"Total overlapping rows: {len(overlap_df):,}")
print(f"Total overlapping unique tracks: {len(unique_overlap):,}")
