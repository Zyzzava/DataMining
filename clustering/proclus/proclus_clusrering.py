import numpy as np
from pyclustering.cluster.proclus import proclus

def apply_proclus(df, unique_texts, tfidf_matrix, k=30, l=5):
    print(f"Applying PROCLUS clustering... on {len(unique_texts)} unique contexts")

    # convert matrix to a dense list of lists
    # does mads computer have enough memory to handle this? if not, we might need to use a sparse representation or batch processing
    dense_data = tfidf_matrix.toarray().tolist()

    # random medioid initialization
    np.random.seed(42)
    initial_medoids = np.random.choice(len(dense_data), k, replace=False).tolist()

    #init and run
    proclus_instance = proclus(dense_data, initial_medoids, l)
    proclus_instance.process()

    #extract clusters 
    clusters = proclus_instance.get_clusters()

    # convert to same data type as kmeans output
    cluster_labels = np.full(len(unique_texts), -1) # Default to -1 just in case PROCLUS drops outliers
    for cluster_id, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            cluster_labels[idx] = cluster_id
    
    
