import numpy as np
from nmf import io, util, metric, algorithm

# Configuration
sample_size = 0.9
epoch = 5
random_state = 0
reduce_scale_yaleB = 4
reduce_scale_orl = 2

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

        # data preprocessing
        V = util.unity_normalise(V)

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
