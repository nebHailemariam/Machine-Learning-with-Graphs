import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from utils import *

# Import necessary utility functions from utils.py
from utils import (
    build_vocab_df,
    preprocess_dataset_df,
    build_vocab_ctf,
    preprocess_dataset_ctf,
)

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.environ["LOKY_MAX_CPU_COUNT"] = "10"  # Set to the desired number of CPU cores

# Check if the correct arguments are provided
if len(sys.argv) < 2:
    print("Usage: ./hw2.sh ctf or ./hw2.sh df")
    sys.exit(1)

argument = sys.argv[1]
print("#########################")
print("K-Means Clustering")
print("Argument received:", argument)

# Load and preprocess data based on the argument
if argument == "df":
    vocab = build_vocab_df(train_data)
    train_features, train_targets = preprocess_dataset_df(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)
else:
    vocab = build_vocab_ctf(train_data)
    train_features, train_targets = preprocess_dataset_ctf(train_data, vocab)
    train_features, train_targets = np.array(train_features), np.array(train_targets)

# Apply Min-Max Normalization
scaler = MinMaxScaler()
train_features_normalized = scaler.fit_transform(train_features)

# Define a range of clusters to experiment with
num_clusters_range = [2, 3, 4, 5]

for num_clusters in num_clusters_range:
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(train_features_normalized)

    # Print cluster labels
    print("Cluster labels for", num_clusters, "clusters:", cluster_labels)

    # Print centroid feature values for each cluster
    centroids = kmeans.cluster_centers_
    print("Centroid feature values for each cluster:", centroids)
    pca = PCA(n_components=2)
    centroids_pca = pca.fit_transform(centroids)
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    # Define colors for each cluster
    colors = ["b", "g", "r", "c", "m"]
    for cluster in centroids_pca:
        plt.scatter(
            cluster[0],
            cluster[1],
            label=f"Cluster {cluster}",
            alpha=0.5,
        )
    plt.title("K-Means Clustering Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
    # Add cluster labels to the data
    train_data_with_clusters = [
        [data, label] for data, label in zip(train_data, cluster_labels)
    ]

    # Apply PCA to reduce dimensionality to 2 dimensions
    pca = PCA(n_components=2)
    train_features_pca = pca.fit_transform(train_features_normalized)

    # Visualize clusters
    plt.figure(figsize=(10, 6))
    for cluster in range(num_clusters):
        plt.scatter(
            train_features_pca[cluster_labels == cluster, 0],
            train_features_pca[cluster_labels == cluster, 1],
            label=f"Cluster {cluster}",
            color=colors[cluster],
            alpha=0.5,
        )
    plt.title("K-Means Clustering Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
