"""NMF algorithm implementation module."""
import numpy as np
from sklearn.decomposition import NMF


def benchmark(V, Yhat):
    """Set up a benchmark model using NMF in scikit-learn.

    Parameters
    ----------
    V: np.ndarray (d, n) where d is the number of pixel
        The contaminated image dataset
    Yhat: np.ndarray (n,)
        The label of images

    Returns
    -------
    W: np.ndarray (d, m)
        The common structure
    H: np.ndarray (m, n)
        The new representation of image data V

    """
    model = NMF(n_components=len(set(Yhat)))
    W = model.fit_transform(V)
    H = model.components_
    return W, H
