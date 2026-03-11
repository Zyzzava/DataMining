import os
import spacy
import pandas as pd
from tqdm import tqdm

# Define your custom stop words here
CUSTOM_STOP_WORDS = {
    "playlist", "music", "song", "songs", "track", "tracks", 
    "mix", "tape", "mixtape", "vol", "volume", "part", "pt"
}

def remove_stop_words(text, nlp):
    """Removes standard spaCy stop words from a string."""
    if not isinstance(text, str):
        return text
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

def filter_stop_words_step(df, output_parquet):
    """Filters stop words from homogenized playlists and saves the state."""
    if 'filtered_playlist' in df.columns:
        print("\nStop words already filtered. Skipping.")
        return df

    print("\nFiltering stop words from homogenized playlists...")
    
    # Load spaCy without heavy pipelines for speed
    nlp = spacy.load("en_core_web_lg", disable=["parser", "attribute_ruler", "lemmatizer"])
    
    # --- NEW: Add custom stop words to the spaCy vocabulary ---
    for word in CUSTOM_STOP_WORDS:
        nlp.vocab[word].is_stop = True
    # ----------------------------------------------------------

    unique_names = df['homogenized_playlist'].unique()
    stop_word_map = {
        name: remove_stop_words(str(name), nlp) 
        for name in tqdm(unique_names, desc="Stop Word Filtering")
    }
    
    df['filtered_playlist'] = df['homogenized_playlist'].map(stop_word_map)
    df.to_parquet(output_parquet, index=False)
    
    return df