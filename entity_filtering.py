import pandas as pd
import spacy
import time
import sys
from tqdm import tqdm

def setup_knowledge_base(file_path):
    """Loading screen for the heavy Parquet file and Set generation"""
    print(f"\n{'='*50}")
    print(f"INITIALIZING KNOWLEDGE BASE")
    print(f"{'='*50}")
    
    # 1. Load Parquet with a simple spinner-style notice
    print(f"Reading {file_path}...")
    start_time = time.time()
    df_full = pd.read_parquet(file_path)
    print(f"Done! Loaded {len(df_full):,} rows in {time.time() - start_time:.2f}s")

    # 2. Build Lookup Sets with Progress Bars
    # We use unique() first to make the Set building faster
    print("\nExtracting unique Artists...")
    artists = set(tqdm(df_full['artistname'].str.lower().dropna().unique(), desc="Building Artist Set"))
    
    print("\nExtracting unique Tracks...")
    tracks = set(tqdm(df_full['trackname'].str.lower().dropna().unique(), desc="Building Track Set"))

    genres = {
        'rock', 'pop', 'hip hop', 'rap', 'jazz', 'country', 'classical', 
        'metal', 'edm', 'r&b', 'indie', 'dance', 'house', 'techno'
    }
    
    return artists, tracks, genres

def is_contextual_playlist(playlist_name, known_artists, known_tracks, known_genres):
    """The logic remains the same, but we pass the sets in to avoid reloading every time"""
    # --- CONFIGURATION ---
    # Use the large model as requested, but disable components we don't need for speed
    print("Loading spaCy Language Model (en_core_web_lg)...")
    nlp = spacy.load("en_core_web_lg", disable=["parser", "attribute_ruler"])
    
    if not playlist_name or not isinstance(playlist_name, str):
        return False
    
    name_low = playlist_name.lower()
    
    # Exact Match Checks
    if name_low in known_artists or name_low in known_tracks or name_low in known_genres:
        return False
        
    doc = nlp(playlist_name)
    if not doc.ents:
        return True
    
    non_contextual_entities = ['PERSON', 'ORG']
    for ent in doc.ents:
        if ent.label_ in non_contextual_entities:
            # If entity dominates the string
            if len(ent.text) >= (len(playlist_name) * 0.9):
                # Hallucination check
                if any(token.pos_ in ['ADJ', 'VERB'] for token in ent):
                    return True 
                else:
                    return False
    return True

def main():
    # 1. Run the Setup Loading Screen
    try:
        known_artists, known_tracks, known_genres = setup_knowledge_base('spotify_final_healed.parquet')
    except FileNotFoundError:
        print("Error: Parquet file not found. Please ensure the file path is correct.")
        return

    # 2. Load the data you want to filter (using mock for example)
    mock_data = {
        'homogenized_playlist': [
            'summer party 2015', 'workout track', 'michael jackson', 
            'the beatles', 'lofi hip hop for studying', 'Kendrick Lamar',
            'cooking dinner in italy', 'Metallica - Black Album', 'chill vibes'
        ] * 100  # Multiplying just to show the progress bar in action
    }
    mdf = pd.DataFrame(mock_data)

    # 3. Processing Loading Screen
    print(f"\n{'='*50}")
    print(f"EVALUATING {len(mdf):,} PLAYLISTS")
    print(f"{'='*50}")
    
    # Use tqdm with pandas apply
    tqdm.pandas(desc="Filtering Progress")
    mdf['keep_playlist'] = mdf['homogenized_playlist'].progress_apply(
        is_contextual_playlist, 
        args=(known_artists, known_tracks, known_genres)
    )

    filtered_df = mdf[mdf['keep_playlist'] == True].copy()
    
    print(f"\n{'='*50}")
    print(f"FILTERING COMPLETE")
    print(f"Kept {len(filtered_df):,} out of {len(mdf):,} playlists.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()