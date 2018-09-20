"""NMF algorithm implementation module."""
import numpy as np
from tqdm import tqdm


def multiplication_euclidean(V, r,niter,min_error1):
    """Implement NMF with multiplication Euclidean algorithm."""
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)*255
    H=np.random.rand(r,n)*255
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


def multiplication_divergence(V, r,niter,min_error):
    """Implement NMF with multiplication KL divergence algorithm."""
    m=V.shape[0]
    n=V.shape[1]
    W=np.random.rand(m,r)*255
    H=np.random.rand(r,n)*255
    Oness=np.ones((m, n))
    error = []
    for i in tqdm(range(niter)):
        W=W/(Oness@H.T)*(V/(W@H)@H.T)
        H=H/(W.T@Oness)*(W.T@(V/(W@H)))
        #calculate the distance between iteration
        e = np.sum(V*np.log((V+1e-13)/(W@H))-V+W@H)/V.size
        error.append(e)
        #stop iteration if distance less than min_error
        if e < min_error:
            break
    return W, H, error
