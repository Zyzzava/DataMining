import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import pickle
import seaborn as sns
import scipy.sparse
from scipy.sparse import issparse
from wordcloud import WordCloud

# Create directory for saving plots
OUTPUT_DIR = "clustering/tf_idf_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
This module provides a comprehensive analysis of the TF-IDF matrix derived from the contextual features of playlists"""
def load_tfidf_matrix(tfidf_cache_dir, df, TfidfVectorizer):
    os.makedirs(tfidf_cache_dir, exist_ok=True)

    # Define file paths for the cached objects
    matrix_path = os.path.join(tfidf_cache_dir, "cleaned_tfidf_matrix.npz")
    texts_path = os.path.join(tfidf_cache_dir, "cleaned_unique_texts.pkl")
    vectorizer_path = os.path.join(tfidf_cache_dir, "vectorizer.pkl")

    # Check if all cached files exist
    if os.path.exists(matrix_path) and os.path.exists(texts_path) and os.path.exists(vectorizer_path):
        print("[INFO] Loading cached, cleaned TF-IDF matrix and unique texts...")
        tfidf_matrix = scipy.sparse.load_npz(matrix_path)
        
        with open(texts_path, 'rb') as f:
            unique_texts = pickle.load(f)
            
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
        print(f"[INFO] Loaded TF-IDF matrix shape: {tfidf_matrix.shape}")

    else:
        print("[INFO] No cache found. Extracting unique contextual features for TF-IDF matrix...")
        contextual_mask = df['is_contextual'] == True
        unique_texts = df[contextual_mask]['expanded_features'].dropna().unique()
        
        print(f"[INFO] Creating TF-IDF matrix for {len(unique_texts):,} unique contexts...")
        # Notice 5678 features maintaining ~80% of the information.
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.95, max_features=5678)
        tfidf_matrix = vectorizer.fit_transform(unique_texts)

        # --- Noise Filtering Block ---
        row_sums = np.squeeze(np.asarray(tfidf_matrix.sum(axis=1)))
        non_empty_mask = row_sums > 0
        
        dropped_count = len(unique_texts) - non_empty_mask.sum()
        if dropped_count > 0:
            print(f"[WARNING] Dropping {dropped_count:,} contexts that became empty after TF-IDF filtering (Noise).")
        
        # Apply the filter
        unique_texts = unique_texts[non_empty_mask]
        tfidf_matrix = tfidf_matrix[non_empty_mask]
        
        # --- Save to Cache ---
        print(f"[INFO] Saving cleaned TF-IDF matrix and objects to '{tfidf_cache_dir}'...")
        scipy.sparse.save_npz(matrix_path, tfidf_matrix)
        
        with open(texts_path, 'wb') as f:
            pickle.dump(unique_texts, f)
            
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)


def plot_wordcloud(tfidf_matrix, vectorizer):
    print("Generating WordCloud...")
    feature_names = vectorizer.get_feature_names_out()
    avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    
    weights_dict = dict(zip(feature_names, avg_weights))
    
    wordcloud = WordCloud(
        width=1600, 
        height=800, 
        background_color='white',
        colormap='viridis',
        max_words=200
    ).generate_from_frequencies(weights_dict)
    
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(f"{OUTPUT_DIR}/wordcloud.png")
    plt.close()

def get_top_tfidf_features(tfidf_matrix, vectorizer, top_n=25):
    feature_names = vectorizer.get_feature_names_out()
    avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'avg_tfidf': avg_weights
    }).sort_values(by='avg_tfidf', ascending=False)
    return feature_df

def plot_top_features(tfidf_matrix, vectorizer, top_n=30):
    print(f"Generating Top {top_n} Features plot...")
    top_df = get_top_tfidf_features(tfidf_matrix, vectorizer, top_n).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_df, 
        x='avg_tfidf', 
        y='feature', 
        hue='feature', 
        palette='viridis', 
        legend=False
    )
    plt.title(f"Top {top_n} Features by Average TF-IDF Score")
    plt.xlabel("Average TF-IDF Weight")
    plt.ylabel("Word/Artist")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_features.png")
    plt.close()

def plot_weight_distribution(tfidf_matrix):
    print("Generating Weight Distribution plot...")
    weights = tfidf_matrix.data if issparse(tfidf_matrix) else tfidf_matrix[tfidf_matrix > 0]
        
    plt.figure(figsize=(10, 6))
    sns.histplot(weights, bins=50, kde=True, color='skyblue')
    plt.title("Distribution of Non-Zero TF-IDF Weights")
    plt.xlabel("TF-IDF Weight")
    plt.ylabel("Frequency")
    plt.yscale('log') # Log scale handles the massive imbalance in sparse text data
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/weight_distribution.png")
    plt.close()

def plot_document_frequency(tfidf_matrix):
    """
    Plots the Document Frequency (DF) of features (how many playlists a word appears in).
    This helps justify the min_df and max_df thresholds.
    """
    print("Generating Document Frequency plot...")
    # Count non-zero entries per column to get DF
    doc_freqs = np.asarray((tfidf_matrix > 0).sum(axis=0)).ravel()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(doc_freqs, bins=50, color='coral', kde=False)
    plt.title("Document Frequency of Features (Zipf's Law)")
    plt.xlabel("Number of Contexts (Playlists) the Feature Appears In")
    plt.ylabel("Number of Features")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/document_frequency.png")
    plt.close()
    
    return doc_freqs

def plot_cumulative_importance(tfidf_matrix):
    """
    Calculates the cumulative sum of TF-IDF weights to determine optimal max_features.
    Returns the feature counts required to hit 80%, 90%, and 95% total variance.
    """
    print("Generating Cumulative Importance plot...")
    avg_weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    sorted_weights = np.sort(avg_weights)[::-1]
    
    # Calculate cumulative percentage
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_percent = (cumulative_weights / cumulative_weights[-1]) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_percent, linewidth=2, color='darkorange')

    # Find specific percentage cutoffs
    thresholds = [80, 90, 95]
    cutoffs = {}
    colors = ['red', 'green', 'blue']
    
    for threshold, color in zip(thresholds, colors):
        # Find index where we first hit the threshold
        idx = np.argmax(cumulative_percent >= threshold)
        features_needed = idx + 1
        cutoffs[threshold] = features_needed
        
        plt.axvline(x=features_needed, color=color, linestyle='--', 
                    label=f'{threshold}% Information ({features_needed:,} features)')
        plt.axhline(y=threshold, color=color, linestyle=':', alpha=0.5)

    plt.title("Cumulative Importance of Features (Scree Plot)")
    plt.xlabel("Number of Features (Sorted by Importance)")
    plt.ylabel("Cumulative Importance (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cumulative_importance.png")
    plt.close()
    
    return cutoffs

def generate_comprehensive_report(tfidf_matrix, cutoffs, doc_freqs):
    """
    Generates a detailed text report with concrete guidelines for max_features.
    """
    n_elements = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
    n_nonzero = tfidf_matrix.nnz if issparse(tfidf_matrix) else np.count_nonzero(tfidf_matrix)
    sparsity = (1 - (n_nonzero / n_elements)) * 100
    
    print("\n--- TF-IDF Matrix Statistics & Guidelines ---")
    report = f"""TF-IDF MATRIX ANALYSIS REPORT
    =============================
    Matrix Shape: {tfidf_matrix.shape}
    Total Possible Elements: {n_elements:,}
    Non-Zero Elements: {n_nonzero:,}
    Matrix Sparsity: {sparsity:.4f}%

    DOCUMENT FREQUENCY (DF) STATS
    -----------------------------
    Mean Document Frequency: {np.mean(doc_freqs):.1f} playlists
    Median Document Frequency: {np.median(doc_freqs):.1f} playlists
    Max Document Frequency: {np.max(doc_freqs)} playlists

    MAX_FEATURES RECOMMENDATIONS
    -----------------------------
    To speed up K-Means or BIRCH without losing the core structure of your data, 
    you can drop the "long tail" of obscure words.

    * To retain 80% of total information: set max_features={cutoffs[80]}
    * To retain 90% of total information: set max_features={cutoffs[90]}
    * To retain 95% of total information: set max_features={cutoffs[95]}

    Guideline: 
    If BIRCH/K-Means is running out of memory or is too slow, use the {cutoffs[80]} 
    feature threshold. If it runs fine but clustering metrics are poor, increase 
    towards the {cutoffs[95]} threshold to capture finer nuances.
    """
    print(report)
    with open(f"{OUTPUT_DIR}/analysis_report.txt", "w") as f:
        f.write(report)

def run_full_tfidf_analysis(tfidf_matrix, vectorizer):
    print("\nStarting Comprehensive TF-IDF Analysis...")
    
    # Basic Visualizations
    plot_top_features(tfidf_matrix, vectorizer)
    plot_weight_distribution(tfidf_matrix)
    plot_wordcloud(tfidf_matrix, vectorizer)
    
    # Advanced Statistical Analysis
    doc_freqs = plot_document_frequency(tfidf_matrix)
    cutoffs = plot_cumulative_importance(tfidf_matrix)
    
    # Generate Guidelines
    generate_comprehensive_report(tfidf_matrix, cutoffs, doc_freqs)
    print(f"\nAnalysis complete. Check '{OUTPUT_DIR}/analysis_report.txt' for max_features recommendations.")