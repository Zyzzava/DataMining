import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from graph.knn.knn_graph import KNNGraph
from preprocessing.preprocessor import FULLY_PROCESSED_PARQUET

def run_integration_test():
    # Load your processed data
    df = pd.read_parquet(FULLY_PROCESSED_PARQUET)
    
    # Filter unique texts to avoid coordinate stacking
    contextual_mask = df['is_contextual'] == True
    unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()
    
    # Use a smaller test set for cleaner visualization
    sample_size = min(300, len(unique_texts))
    test_samples = np.random.choice(unique_texts, size=sample_size, replace=False)

    # Standard TF-IDF config
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(test_samples)

    # Build and visualize
    knn = KNNGraph(k_neighbors=4) 
    knn.build_graph(tfidf_matrix, test_samples)
    knn.visualize_improved()

if __name__ == "__main__":
    run_integration_test()