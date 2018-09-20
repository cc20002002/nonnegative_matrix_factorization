"""Implement metrics of NMF."""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score


def eval_rre(Vhat, W, H):
    """Evaluate relative reconstruction error."""
    rre = np.linalg.norm(Vhat - W.dot(H)) / np.linalg.norm(Vhat)
    return rre


def eval_acc(Yhat, Ypred):
    """Evaluate accuracy of actual label versus Kmeans cluster label."""
    acc = accuracy_score(Yhat, Ypred)
    return acc


def eval_nmi(Yhat, Ypred):
    """Evaluate normalised mutual information score."""
    nmi = normalized_mutual_info_score(Yhat, Ypred)
    return nmi
