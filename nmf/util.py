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


def draw_image(V, subVhat, V_noise, sample_index):
    """Draw image before and after adding noise."""
    img_size = [x // reduce_scale_orl for x in orl_img_size]
    reshape_size = [img_size[1], img_size[0]]
    V_processed = util.unity_normalise(V)
    pl.figure(figsize=(10,6))
    pl.subplot(221)
    pl.imshow(subVhat[:, sample_index].reshape(reshape_size), cmap=cmap)
    pl.title('Image(Original)')
    pl.subplot(222)
    pl.imshow(V_noise[:, sample_index].reshape(reshape_size), cmap=cmap)
    pl.title('Noise')
    pl.subplot(223)
    pl.imshow(V[:, sample_index].reshape(reshape_size), cmap=cmap)
    pl.title('Image(Noise)')
    pl.subplot(224)
    pl.imshow(V_processed[:, sample_index].reshape(reshape_size), cmap=cmap)
    pl.title('Image(Preprocessed)')
    pl.show()
