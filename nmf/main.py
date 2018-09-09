import numpy as np
import os
from nmf import io, util, metric, algorithm
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


def main():
    """Run NMF on CroppedYaleB and ORL dataset."""
    if os.name == 'nt':
        train("..\\data\\ORL")
        # train("data/CroppedYaleB")
    else:
        train("data/ORL")
        # train("data/CroppedYaleB")


def train(data_name):
    """Run NMF on data stored in data_name."""
    # load ORL dataset
    print("==> Load {} dataset...".format(data_name))
    Vhat, Yhat = io.load_data(data_name, reduce_scale_orl)

    n = len(Yhat)
    size = int(n * sample_size)
    rre, acc, nmi = [], [], []

    for i in range(epoch):
        print("Epoch {}...".format(i + 1))
        # sample 90% of samples
        index = np.random.choice(np.arange(n), size, replace=False)
        subVhat, subYhat = Vhat[:, index], Yhat[index]

        # DONE: we might need to implement other noise
        # add noise (Gaussian noise)
        V_noise = np.random.normal(0, 1, subVhat.shape) #* np.sqrt(subVhat)
        V = subVhat + V_noise        
        V2 = np.random.poisson(subVhat)
        V_noise2 = V2-subVhat
        V[(V < 0)]=0
        #import IPython;
        #IPython.embed()
        #print(np.linalg.norm(np.sort(V_noise2.ravel()) - np.sort(V_noise.ravel())) / np.linalg.norm(V_noise2))
        #check whether similar
        if i == 0:
            # draw image before and after adding noise
            img_size = [x // reduce_scale_orl for x in orl_img_size]
            reshape_size = [img_size[1], img_size[0]]
            V_processed = util.unity_normalise(V)
            pl.figure(figsize=(10,6))
            pl.subplot(321)
            pl.imshow(subVhat[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Image(Original)')
            pl.subplot(322)
            pl.imshow(V_noise[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Noise')
            pl.subplot(323)
            pl.imshow(V[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Image(Noise)')
            pl.subplot(324)
            pl.imshow(V_processed[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Image(Preprocessed)')
            pl.subplot(325)
            pl.imshow(V_noise2[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Noise')
            pl.subplot(326)
            pl.imshow(V2[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Image(Noise)')
            pl.show()

        # TODO: use our algorithm here
        # apply NMF algorithm (benchmark) for now
        r = np.unique(Yhat).shape[0]
        W, H = algorithm.multiplication_divergence(V, r)
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
