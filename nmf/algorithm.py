"""NMF algorithm implementation module."""
import numpy as np
from sklearn.decomposition import NMF


def benchmark(V, r):
    """Set up a benchmark model using NMF in scikit-learn.

    Parameters
    ----------
    V: np.ndarray (d, n) where d is the number of pixel
        The contaminated image dataset
    r: integer
        The number of basis (classes)

    Returns
    -------
    W: np.ndarray (d, m)
        The common structure
    H: np.ndarray (m, n)
        The new representation of image data V

    """
    model = NMF(n_components=r)
    W = model.fit_transform(V)
    H = model.components_
    return W, H
