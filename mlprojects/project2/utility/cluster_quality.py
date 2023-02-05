# Copyright © 2023 "Bronte" Sihan Li

"""
This module contains helper functions to calculate the cluster quality for
a given dataset.
"""

import numpy as np
from sklearn.cluster import KMeans


# compute cluster quality for datasets
def get_representation_error(data, kmeans: KMeans):
    """
    calculate representation error (sum of squared error)
    """
    # get cluster centers
    centers = kmeans.cluster_centers_
    # get cluster labels
    labels = kmeans.labels_
    # calculate the sum of squared euclidean distance (L2 norm) between each point and its cluster center
    return np.sum((np.linalg.norm(data - centers[labels], axis=1)) ** 2)


def cluster_quality(data, kmeans: KMeans, metric: str = 'elbow'):
    """
    calculate cluster quality for a given dataset using the elbow method, MDL or Ray-Turi
    """
    # get the representation error
    rep_error = get_representation_error(data, kmeans)
    # get the number of clusters
    n_clusters = kmeans.n_clusters
    # get the number of data points
    n_points = data.shape[0]
    # get number of features
    p = data.shape[1]
    # calculate the cluster quality
    if metric == 'elbow':
        assert n_clusters > 2, 'Number of clusters must be greater than 2'
        # calculate sse for k-1 and k-2 clusters
        kmeans_1 = KMeans(n_clusters=n_clusters - 1, random_state=0).fit(data)
        kmeans_2 = KMeans(n_clusters=n_clusters - 2, random_state=0).fit(data)
        # calculate diff for k
        diff = (n_clusters - 1) ** (2 / p) * get_representation_error(
            data, kmeans_1
        ) - n_clusters ** (2 / p) * rep_error
        # calculate diff for k-1
        diff_1 = (n_clusters - 2) ** (2 / p) * get_representation_error(
            data, kmeans_2
        ) - (n_clusters - 1) ** (2 / p) * get_representation_error(data, kmeans_1)
        return diff / diff_1
    elif metric == 'mdl':
        return rep_error + (n_clusters * np.log2(n_points)) / 2
    elif metric == 'ray-turi':
        # Calculate intra-cluster distances, this is the same as the representation error
        intra = rep_error
        # Calculate inter-cluster distances
        centers = kmeans.cluster_centers_
        distances = [
            (np.linalg.norm(centers[i] - centers[j])) ** 2
            for i in range(n_clusters)
            for j in range(i + 1, n_clusters)
        ]
        inter = np.min(distances)
        return intra / inter
    else:
        raise ValueError('Unknown metric of cluster quality')
