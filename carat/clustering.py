# encoding: utf-8
# pylint: disable=C0103
# pylint: disable=too-many-arguments
"""
Clustering
==========

Clustering and manifold learning
--------------------------------
.. autosummary::
    :toctree: generated/

    rhythmic_patterns
    manifold_learning
"""


from sklearn import cluster
from sklearn import manifold

__all__ = ['rhythmic_patterns', 'manifold_learning']

def rhythmic_patterns(data, n_clusters=4, method='kmeans'):
    """Clustering of rhythmic patterns from feature map.

    Parameters
    ----------
    data  :  np.ndarray
        feature map
    n_clusters : int
        number of clusters
    method : str
        clustering method

    Returns
    -------
    c_labs : np.ndarray
        cluster labels for each data point
    c_centroids : np.ndarray
        cluster centroids
    c_method : sklearn.cluster
        sklearn cluster method object

    Notes
    --------
    Based on the feature map clustering analysis introduced in [1].

    References
    --------
    .. [1] Rocamora, Jure, Biscainho
           "Tools for detection and classification of piano drum patterns from candombe recordings."
           9th Conference on Interdisciplinary Musicology (CIM),
           Berlin, Germany. 2014.

    See Also
    --------
    sklearn.cluster.KMeans
    """

    if method == 'kmeans':
        # initialize k-means algorithm
        c_method = cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)

        # cluster data using k-means
        c_method.fit(data)

        # cluster centroids
        c_centroids = c_method.cluster_centers_

        # predict cluster for each data point
        c_labs = c_method.predict(data)

    else:
        raise AttributeError("Clustering method not implemented.")

    return c_labs, c_centroids, c_method


def manifold_learning(data, method='isomap', n_neighbors=7, n_components=3):
    """Manifold learning for dimensionality reduction of rhythmic patterns data.


    Parameters
    ----------
    data : np.array
        feature map
    method : (check)
        (check)
    n_neighbors : int
        number of neighbors for each dat point
    n_components : int
        number of coordinates for the manifold

    Returns
    -------
    embedding : np.array
        lower-dimensional embedding of the data

    Notes
    --------
    Based on the dimensionality reduction for rhythmic patterns introduced in [1].

    References
    --------
    .. [1] Rocamora, Jure, Biscainho
           "Tools for detection and classification of piano drum patterns from candombe recordings."
           9th Conference on Interdisciplinary Musicology (CIM),
           Berlin, Germany. 2014.

    """

    if method == 'isomap':
        # fit manifold from data using isomap algorithm
        method = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components).fit(data)

        # transform data to low-dimension representation
        embedding = method.transform(data)

    else:
        raise AttributeError("Manifold learning method not implemented.")

    return embedding
