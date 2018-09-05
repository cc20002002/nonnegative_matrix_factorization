import numpy as np
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

        # TODO: we might need to implement other noise
        # add noise (Gaussian noise)
        V_noise = np.random.rand(*subVhat.shape) * 40
        V = subVhat + V_noise

        if i == 0:
            # draw image before and after adding noise
            img_size = [x // reduce_scale_orl for x in orl_img_size]
            reshape_size = [img_size[1], img_size[0]]
            V_processed = util.unity_normalise(V)
            pl.figure(figsize=(10,6))
            pl.subplot(221)
            pl.imshow(subVhat[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Image(Original)')
            pl.subplot(222)
            pl.imshow(V_noise[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Noise')
            pl.subplot(223)
            pl.imshow(V[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Image(Noise)')
            pl.subplot(224)
            pl.imshow(V_processed[:, sample_index].reshape(reshape_size), cmap=cmap)
            pl.title('Image(Preprocessed)')
            pl.show()
            # draw image before and after normalisation

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
