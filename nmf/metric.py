"""Implement metrics of NMF."""
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score


def eval_rre(Vhat, W, H):
    """Evaluate relative reconstruction error.

    Parameters
    ----------
    Vhat: np.ndarray (d, n) where d is the number of pixel
        The clear image dataset
    W: np.ndarray (d, m)
        The common structure
    H: np.ndarray (m, n)
        The new representation of image data V

    Returns
    -------
    rre: float
        The relative reconstruction error

    """
    rre = np.linalg.norm(Vhat - W.dot(H)) / np.linalg.norm(Vhat)
    return rre


def eval_acc(Yhat, Ypred):
    """Evaluate accuracy of actual label versus Kmeans cluster label.

    Parameters
    ----------
    Yhat: np.ndarray (n,)
        The acutal label of images
    Ypred: np.ndarray (n,)
        The Kmeans predicted label of images

    Returns
    -------
    acc: float
        The accuray of label predictions

    """
    acc = accuracy_score(Yhat, Ypred)
    return acc


def eval_nmi(Yhat, Ypred):
    """Evaluate normalised mutual information score.

    Parameters
    ----------
    Yhat: np.ndarray (n,)
        The acutal label of images
    Ypred: np.ndarray (n,)
        The Kmeans predicted label of images

    Returns
    -------
    nmi: float
        The normalized_mutual_info_score.

    """
    nmi = normalized_mutual_info_score(Yhat, Ypred)
    return nmi


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
