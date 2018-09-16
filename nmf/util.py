"""Utility functions for NMF."""
import numpy as np
from collections import Counter
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans


def unity_normalise(data):
    """Apply unity normalisation to preprocess non-negative image data.

    Parameters
    ----------
    data: np.ndarray (d, n) where d is the number of pixel
        The image dataset.

    Returns
    -------
    normalised_data: np.ndarray(n, d)
        The unity-normalised image data. (still non-negative)

    """
    col_min = np.min(data, axis=0)
    col_max = np.max(data, axis=0)
    normalised_data = (data - col_min) / (col_max - col_min)
    return normalised_data


def assign_cluster_label(X, Y):
    """Cluster X based on number of unique labels in Y.

    Parameters
    ----------
    X: np.ndarray (d, n) where d is the number of pixel
        The contaminated image dataset
    Y: np.ndarray (n,)
        The label of images

    Returns
    -------
    Y_pred: np.ndarray(n, )
        The Kmeans predicted clustering label

    """
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def error_vs_iter(error, niter, algo_name, path):
    """Plot error versus iteration."""
    pl.figure()
    pl.plot(np.arange(niter), error)
    pl.xlabel("Iteration")
    pl.ylabel("Error")
    pl.title("{} Training Error versus {} Iteration".format(algo_name, niter))
    pl.savefig(path)
