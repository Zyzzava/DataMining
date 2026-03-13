import numpy as np
from sklearn.cluster import Birch
from tqdm import tqdm


def apply_birch(df, unique_texts, tfidf_matrix, k=34):
    print(f"Applying BIRCH clustering... on {len(unique_texts)} unique contexts")

    birch = Birch(
        threshold=0.5,       # Increased to reduce tree size
        n_clusters=k,
        branching_factor=50  # Default is usually fine if threshold is high enough
    )

    # progress bar for fitting
    print("Fitting BIRCH model...")
    cluster_labels = birch.fit_predict(tfidf_matrix)
    print(f"BIRCH clustering completed. Found {len(set(cluster_labels))} clusters.")

    # Progress bar
    print("Building cluster mapping...")
    cluster_mapping = {
        text: label for text, label in tqdm(zip(unique_texts, cluster_labels), 
        total=len(unique_texts), desc="Zipping labels")
    }

    # Progress bar
    print(f"Broadcasting labels to {len(df):,} rows...")
    tqdm.pandas(desc="Mapping Clusters")
    df[f'birch_cluster_{k}'] = df['expanded_features'].progress_map(cluster_mapping.get)    

    print(f"Added 'birch_cluster_{k}' labels to the DataFrame.")
    return df