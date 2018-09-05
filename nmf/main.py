import numpy as np
from nmf import io, util, metric, algorithm

# Configuration
reduce_scale_yaleB = 4
reduce_scale_orl = 2

def main():
    """Run NMF on CroppedYaleB and ORL dataset."""
    train("data/ORL")
    train("data/CroppedYaleB")


def train(data_name):
    """Run NMF on data stored in data_name."""
    # load ORL dataset
    print("==> Load {} dataset...".format(data_name))
    Vhat, Yhat = io.load_data(data_name, reduce_scale_orl)

    # add noise
    V_noise = np.random.rand(*Vhat.shape) * 40
    V = Vhat + V_noise

    # data preprocessing
    V = util.unity_normalise(V)

    # apply NMF algorithm (benchmark)
    W, H = algorithm.benchmark(V, Yhat)
    Ypred = util.assign_cluster_label(H.T, Yhat)

    # evaluate metrics
    rre = metric.eval_rre(Vhat, W, H)
    acc = metric.eval_acc(Yhat, Ypred)
    nmi = metric.eval_nmi(Yhat, Ypred)
    print("Evaluate RRE = ", rre)
    print("Evaluate ACC = ", acc)
    print("Evaluate NMI = ", nmi)


if __name__ == "__main__":
    main()
