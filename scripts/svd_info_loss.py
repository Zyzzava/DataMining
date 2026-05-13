import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import TruncatedSVD

def plot_svd_variance(tfidf_matrix, random_seed):
    # Create folder in clustering/reports called 'SVD' if it doesn't exist
    if not os.path.exists("clustering/reports/SVD"):
        os.makedirs("clustering/reports/SVD")

    # if already exit, just plot 
    if os.path.exists("clustering/reports/SVD/svd_variance_plot.png"):
        print("\n[INFO] SVD variance plot already exists. Loading from file...")
        img = plt.imread("clustering/reports/SVD/svd_variance_plot.png")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return
    # 1. Initialize SVD with a large number of components to test
    # (Checking up to 1500 components to see the variance curve)
    test_svd = TruncatedSVD(n_components=1500, random_state=random_seed)

    # 2. Fit the model to our existing TF-IDF matrix
    print("Fitting SVD on 1500 components")
    test_svd.fit(tfidf_matrix)

    # 3. Calculate the cumulative sum of the variance explained
    cumulative_variance = np.cumsum(test_svd.explained_variance_ratio_)

    # 4. Plot the Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--', markersize=1)

    # Draw reference lines for common thresholds
    plt.axhline(y=0.80, color='r', linestyle='-', label='80% Variance Threshold')
    plt.axhline(y=0.90, color='g', linestyle='-', label='90% Variance Threshold')

    plt.title('Truncated SVD: Cumulative Explained Variance')
    plt.xlabel('Number of Components (Latent Features)')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Save the plot for future reference
    plt.savefig("clustering/reports/SVD/svd_variance_plot.png", dpi=300, bbox_inches='tight')
    plt.show()