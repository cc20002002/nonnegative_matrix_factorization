"""Utility functions for NMF."""
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans


def assign_cluster_label(X, Y):
    """Cluster X based on number of unique labels in Y."""
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def error_vs_iter(error, niter, algo_name, path):
    """Plot error versus iteration."""
    dataname = path.split(os.sep)[1].split("-")[-1]
    x = min(len(error), niter)
    pl.figure(figsize=(10, 6))
    pl.plot(np.arange(x), np.log(error))
    newticks = np.around(np.exp(pl.yticks()[0]), decimals=2)
    pl.yticks(pl.yticks()[0], newticks)
    pl.xlabel("Iteration")
    pl.ylabel("Error (Log Scale)")
    pl.title("{} {} Training Error versus {} Iteration"
             .format(dataname, algo_name, niter))
    pl.savefig(path)
