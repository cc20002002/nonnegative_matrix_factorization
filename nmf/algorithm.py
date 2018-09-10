"""NMF algorithm implementation module."""
import numpy as np
from sklearn.decomposition import NMF

niter = 1000


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


def truncated_cauchy(V, r):
    """Set up an truncated cauchy NMF.

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
    gamma = 2.
    d, n = V.shape
    import IPython; IPython.embed()
    W = np.zeros((d, r))
    H = np.zeros((r, n))
    for t in range(niter):
        E = V - W.dot(H)
        Q = 1. / (1 + (E / gamma) ** 2)
        


def multiplication_euclidean(V, r):
    """Set up a benchmark model using NMF in scikit-learn.

    Parameters
    ----------
    V: np.ndarray (d, n) where d is the number of pixel
        The contaminated image dataset
    r: an integer, the number of basis (classes)

    Returns
    -------
    W: np.ndarray (d, m)
        The common structure
    H: np.ndarray (m, n)
        The new representation of image data V

    """
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)
    H=np.random.rand(r,n)
    for i in range(1000):
        H=H*(W.T@V)/(W.T@W@H)
        W=W*(V@H.T)/(W@H@H.T)
    return W, H


def multiplication_divergence(V, r):
    """Set up a benchmark model using NMF in scikit-learn.

    Parameters
    ----------
    V: np.ndarray (d, n) where d is the number of pixel
        The contaminated image dataset
    r: an integer, the number of basis (classes)

    Returns
    -------
    W: np.ndarray (d, m)
        The common structure
    H: np.ndarray (m, n)
        The new representation of image data V

    """
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)
    H=np.random.rand(r,n)
    for i in range(4000):
        VWH = V/(W @ H)
        Numerator1=W.T@VWH
        H=H*Numerator1/np.sum(W,0).reshape(r,1)
    	#only when I looked https://au.mathworks.com/matlabcentral/answers/316708-non-negative-matrix-factorization
    	#I realised I need to calculate VWH again during the update.. waste me 1 hour..
        VWH = V / (W @ H)
        Numerator2=VWH@H.T
        W = W * Numerator2 / np.sum(H, 1).reshape(1,r)
    return W, H
