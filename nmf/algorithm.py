"""NMF algorithm implementation module."""
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF
from numba import vectorize, float64, int64, cuda, jit


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
    return W, H, None


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

def multiplication_euclidean(V, r,niter,min_error1):
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
    print("Multiplication Euclidean Training...")
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)
    H=np.random.rand(r,n)

    error = []
    for i in tqdm(range(niter)):
        H=H*(W.T@V)/(W.T@W@H)
        W=W*(V@H.T)/(W@H@H.T)

        #calculate the distance between iteration
        e = np.sum((V - W@H)**2)/V.size
        error.append(e)
        #stop iteration if distance less than min_error
        if e < min_error1:
            print("iterated: ", i, "times",'error:',e)
            break
    return W, H, error

def multiplication_euclidean_regularized(V, r,niter,min_error1,regu=10,mu=0,nu=0):
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
    print("Multiplication Euclidean regularized Training...")
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)
    H=np.random.rand(r,n)
    #W -= W.min()
    #W = W / W.max()
    error = []
    for i in tqdm(range(niter)):
        
        W=W*(V@H.T)/(W@H@H.T+mu*W)
        
        #W = W / np.linalg.norm(W, axis=1).T.reshape(m,-1)
        H=H*(W.T@V)/(W.T@W@H+regu+nu*H)
        #calculate the distance between iteration
        e = np.sum((V - W@H)**2)/V.size
        error.append(e)
        #stop iteration if distance less than min_error
        if e < min_error1:
            print("iterated: ", i, "times",'error:',e)
            break
    return W, H, error



#@jit('float64[:](float64[:],int64)',nopython=True)
def multiplication_divergence(V, r,niter,min_error):
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
    print("Multiplication KL Divergence Training...")
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)
    H=np.random.rand(r,n)
    #VWH = np.zeros(V.size)
    Oness=np.ones((m, n))
    error = []
    for i in tqdm(range(niter)):
        W=W/(Oness@H.T)*(V/(W@H)@H.T)
        
        
        #only when I looked https://au.mathworks.com/matlabcentral/answers/316708-non-negative-matrix-factorization

        H=H/(W.T@Oness)*(W.T@(V/(W@H)))
        #calculate the distance between iteration
        e = np.sum(V*np.log((V+1e-13)/(W@H))-V+W@H)/V.size
        error.append(e)
        #stop iteration if distance less than min_error
        if e < min_error:
            # print("iterated: ", i, "times","error",e)
            break
    return W, H, error

def multiplication_divergence2(V, r,niter,min_error):
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
    print("Multiplication KL Divergence Training...")
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)
    H=np.random.rand(r,n)
    #VWH = np.zeros(V.size)

    error = []
    for i in tqdm(range(niter)):
        VWH = V/(W @ H)
        Numerator1=W.T@VWH
        H=H*Numerator1/np.sum(W,0).reshape(r,1)
        #only when I looked https://au.mathworks.com/matlabcentral/answers/316708-non-negative-matrix-factorization
        VWH = V / (W @ H)
        Numerator2=VWH@H.T
        W = W * Numerator2 / np.sum(H, 1).reshape(1,r)
        #calculate the distance between iteration
        e = np.sum(V*np.log((V+1e-13)/(W@H))-V+W@H)/V.size
        error.append(e)
        #stop iteration if distance less than min_error
        if e < min_error:
            # print("iterated: ", i, "times","error",e)
            break
    return W, H, error
