"""Utility functions for NMF."""
import numpy as np
from sklearn.decomposition import NMF

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


def benchmark(V, Yhat):
    """Run NMF on scikit-learn as a benchmark model.

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
    H = model.components
    return W, H
