import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from tqdm import tqdm

# Download once
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def homogenize_series(series):
    """
    Optimized for millions of rows with a progress bar.
    """
    # 1. Clean the series using vectorized Pandas operations (extremely fast)
    print("Step 1/4: Lowercasing and removing punctuation...")
    clean_series = (series.astype(str)
                    .str.lower()
                    .str.replace(r'[^\w\s]', '', regex=True))

    # 2. Identify Unique Values
    unique_names = clean_series.unique()
    total_uniques = len(unique_names)
    print(f"Step 2/4: Identified {total_uniques:,} unique names (skipping millions of duplicates)")

    # 3. Process only the unique names with a progress bar
    def process_text(text):
        # Explicit check: if it's not a string, or it's 'nan', or it's empty
        if not isinstance(text, str) or text.lower() == 'nan' or not text.strip():
            return ''
        
        try:
            tokens = word_tokenize(text)
            return ' '.join(lemmatizer.lemmatize(t) for t in tokens)
        except Exception:
            # Fallback for any weird encoding issues
            return str(text)

    print("Step 3/4: Tokenizing and Lemmatizing unique names...")
    # Wrap unique_names in tqdm to see the progress bar
    # We ensure each key in the map is treated as a string
    name_map = {str(name): process_text(name) for name in tqdm(unique_names, desc="Homogenizing")}

    # 4. Map the results back
    print("Step 4/4: Mapping results back to 13 million rows...")
    return clean_series.map(name_map)