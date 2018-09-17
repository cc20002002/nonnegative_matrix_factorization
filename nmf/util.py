"""Utility functions for NMF."""
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
    dataname = path.split("/")[1].split("-")[-1]
    x = min(len(error), niter)
    fig = pl.figure()
    ax = pl.subplot(111)
    ax.plot(np.arange(x), np.log(error))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    yticks = ax.get_yticks()
    ax.set_yticks(np.exp(yticks[1:]))
    pl.title("{} {} Training Error versus {} Iteration"
             .format(dataname, algo_name, niter))
    pl.savefig(path)
