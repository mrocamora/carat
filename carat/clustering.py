# encoding: utf-8
# pylint: disable=C0103
# pylint: disable=too-many-arguments
"""Utility functions to deal with audio."""

from sklearn.cluster import KMeans

__all__ = ['rhythmic_patterns']

def rhythmic_patterns(data, n_clusters=4, method='kmeans'):
    """Clustering of rhythmic patterns from feature map.

    Based on the feature map clustering analysis introduced in [1].

    [1] Rocamora, Jure, Biscainho
           "Tools for detection and classification of piano drum patterns from candombe recordings."
           9th Conference on Interdisciplinary Musicology (CIM),
           Berlin, Germany. 2014.

    **Args**:
        - data (numpy array): feature map
        - n_clusters (int): number of clusters
        - method (str): clustering method
    **Returns**:
        - clusters(numpy array):

    **Raises**:
        -
    """

    if method == 'kmeans':
        # initialize k-means algorithm
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)

        # cluster data using k-means
        kmeans.fit(data)

        # predict cluster for each data point
        clusters = kmeans.predict(data)

    else:
        raise AttributeError("Clustering method not implemented.")

    return clusters
