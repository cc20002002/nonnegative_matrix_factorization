"""NMF algorithm implementation module."""
import numpy as np
import numba
from numpy import random
from sklearn.decomposition import NMF
from numba import vectorize, float64, int64, cuda, jit
niter=5000

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

@jit('float64[:,:],int64,float64[:,:],float64[:,:]',nopython=True,nogil=True,cache=True, parallel=True,fastmath=True,target='gpu')
def multiplication_divergence2(V,r,W,H):
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
    m=V.shape
    n=m[1]
    m=m[0]
    VWH = V
    for i in range(niter):
        #t=time.time()
        matmul(W, H, C)
        VWH = V/(W @ H)
        #print(time.time()-t)
        #t=time.time()
        Numerator1=W.T@VWH
        #print(time.time()-t)
        #t=time.time()
        #import IPython; IPython.embed() 
        H=H*Numerator1/np.sum(W,0).reshape(r,1)
        #print(time.time()-t)
        #t=time.time()
        #only when I looked https://au.mathworks.com/matlabcentral/answers/316708-non-negative-matrix-factorization
        VWH = V / (W @ H)
        #print(time.time()-t)
        #t=time.time()
        Numerator2=VWH@H.T
        #print(time.time()-t)
        #t=time.time()
        W = W * Numerator2 / np.sum(H, 1).reshape(1,r)
        #print(time.time()-t)
        #t=time.time()
    
        #calculate the distance between iteration
        e = np.sum(V*np.log((V+1e-13)/(W@H))-V+W@H)/V.size
        #time.time()-t
        #print("iterated: ", i, "times","error",e)
        #stop iteration if distance less than min_error
        #if e < min_error:
        #print("iterated: ", i, "times","error",e)
        #break

V=np.ones((1000,1000))
r=40
t=time.time()
n=V.shape[1]
m=V.shape[0]
W=np.random.rand(m,r)
H=np.random.rand(r,n)
multiplication_divergence2(V, r,W,H)
print(time.time()-t)

t=time.time()
multiplication_divergence(V, r)
print(time.time()-t)