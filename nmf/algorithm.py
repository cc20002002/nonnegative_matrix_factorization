"""NMF algorithm implementation module."""
import numpy as np
from sklearn.decomposition import NMF

niter = 1000
min_error = 0.00001

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

    for i in range(niter):
        H_u=H*(W.T@V)/(W.T@W@H)
        W_u=W*(V@H_u.T)/(W@H_u@H_u.T)

        #calculate the distance between iteration
        e_W = np.sqrt(np.sum((W_u - W)**2, axis = (0,1)))/W.size
        e_H = np.sqrt(np.sum((H_u - H)**2, axis = (0,1)))/H.size

        #stop iteration if distance less than min_error
        if e_W < min_error and e_H < min_error:
            print("iterated: ", i, "times")
            break
        else:
            H = H_u
            W = W_u
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

    for i in range(niter):
        H_old = H
        W_old = W

        VWH = V/(W @ H)
        Numerator1=W.T@VWH
        H=H*Numerator1/np.sum(W,0).reshape(r,1)
    	#only when I looked https://au.mathworks.com/matlabcentral/answers/316708-non-negative-matrix-factorization
        VWH = V / (W @ H)
        Numerator2=VWH@H.T
        W = W * Numerator2 / np.sum(H, 1).reshape(1,r)

        #calculate the distance between iteration
        e_W = np.sqrt(np.sum((W - W_old)**2, axis = (0,1)))/W.size
        e_H = np.sqrt(np.sum((H - H_old)**2, axis = (0,1)))/H.size

        #stop iteration if distance less than min_error
        if e_W < min_error and e_H < min_error:
            print("iterated: ", i, "times")
            break
    return W, H
