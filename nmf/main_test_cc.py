import numpy as np
execfile('io.py')
import util, metric, algorithm
import matplotlib.pyplot as pl

# Configuration
sample_index = 100
sample_size = 0.9
epoch = 5
cmap = pl.cm.gray
random_state = 0
reduce_scale_yaleB = 4
reduce_scale_orl = 3
orl_img_size = (92, 112)
yaleB_img_size = (168, 192)


data_name="../data/ORL"



"""Run NMF on data stored in data_name."""
# load ORL dataset
print("==> Load {} dataset...".format(data_name))
Vhat, Yhat = load_data(data_name, reduce_scale_orl)

n = len(Yhat)
size = int(n * sample_size)
rre, acc, nmi = [], [], []
i=1
print("Epoch {}...".format(i + 1))
# sample 90% of samples
index = np.random.choice(np.arange(n), size, replace=False)
subVhat, subYhat = Vhat[:, index], Yhat[index]

# TODO: we might need to implement other noise
# add noise (Gaussian noise)
V_noise = np.random.rand(*subVhat.shape) * 0
V = subVhat + V_noise


r=40 # number of basis
m=V.shape[0]
n=V.shape[1]
W=np.random.rand(m,r)
H=np.random.rand(r,n)
method=2

# http://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf
if method=1:
    for i in range(1000):
        H=H*(W.T@V)/(W.T@W@H)
        W=W*(V@H.T)/(W@H@H.T)
elif method=2
    for i in range(1000):
        VWH = V/(W @ H)
        Numerator1=W.T@VWH
        H=H*Numerator1/np.sum(W,0).reshape(r,1)
        #only when I looked https://au.mathworks.com/matlabcentral/answers/316708-non-negative-matrix-factorization
        #I realised I need to calculate VWH again during the update.. waste me 1 hour..
        VWH = V / (W @ H)
        Numerator2=VWH@H.T
        W = W * Numerator2 / np.sum(H, 1).reshape(1,r)


np.linalg.norm(V-W@H,2)/np.linalg.norm(V,2)


# TODO: use our algorithm here
# apply NMF algorithm (benchmark) for now
W, H = algorithm.benchmark(V, subYhat)
Ypred = util.assign_cluster_label(H.T, subYhat)

# evaluate metrics
_rre = metric.eval_rre(subVhat, W, H)
_acc = metric.eval_acc(subYhat, Ypred)
_nmi = metric.eval_nmi(subYhat, Ypred)
print("RRE = {}".format(_rre))
print("ACC = {}".format(_acc))
print("NMI = {}".format(_nmi))
rre.append(_rre)
acc.append(_acc)
nmi.append(_nmi)

# performace over epochs
print("Over {} epochs, average RRE = {}".format(epoch, np.mean(rre)))
print("Over {} epochs, average ACC = {}".format(epoch, np.mean(acc)))
print("Over {} epochs, average NMI = {}".format(epoch, np.mean(nmi)))


if __name__ == "__main__":
    main()
