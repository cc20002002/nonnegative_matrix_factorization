import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from nmf import io, util, metric, algorithm, noise
from util import error_vs_iter
import matplotlib.pyplot as pl
import multiprocessing
from itertools import repeat
import time

# Configuration
sample_size = 0.9
epoch = 1
random_state = 0
multi_start_flag = 0

scale = { "ORL": 3, "CroppedYaleB": 4}
img_size = {"ORL": (92, 112), "CroppedYaleB": (168, 192)}

niter = {
    "Multiplication KL Divergence": 1200,
    "Multiplication Euclidean": 500,
}

min_error = {
    "Multiplication KL Divergence": 2.3,
    "Multiplication Euclidean": 200,
}
model = {
    "Multiplication KL Divergence": algorithm.multiplication_divergence,
    "Multiplication Euclidean": algorithm.multiplication_euclidean,
}

Noise = {
    "No Noise": noise.identity,
    "Poisson": noise.possion,
    "Normal": noise.normal,
    "Salt and Pepper": noise.salt_and_pepper,
}

# fix seed
rnd = np.random.RandomState(random_state)


def main():
    """Run NMF on CroppedYaleB and ORL dataset."""
    argvs = sys.argv
    message = "Please choose one of the two datasets: 'orl' or 'croppedYale'"
    # if no command line argument, then use ORL dataset by default
    if len(argvs) < 2:
        dataname = "orl"
    else:
        assert argvs[1] in ["orl", "croppedYale"], message
        dataname = argvs[1]
    # make a folder with generated time to save results
    folder = datetime.now().strftime("%Y-%m-%d-%H-%M")
    folder = os.path.join("results", folder + "-" + dataname)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # choose dataset to train with
    if dataname == "orl":
        train(".." + os.sep + "data" + os.sep + "ORL", folder)
    else:
        train(".." + os.sep + "data" + os.sep + "CroppedYaleB", folder)


def one_simulation(i,Vhat,Yhat,n,size,metrics,folder):
    # sample 90% of samples
    index = rnd.choice(np.arange(n), size, replace=False)
    subVhat, subYhat = Vhat[:, index], Yhat[index]
    for noise_fun in Noise:
        # add noise
        V, V_noise = Noise[noise_fun](subVhat)
        n_samples = len(Vhat)
        # find number of clusters
        r = np.unique(Yhat).shape[0]
        # loop through different models
        for name, algo in model.items():
            name2 = name
            name = name+' '+noise_fun
            if multi_start_flag == 1:
                # multi-start with various initial values
                ncpu = os.cpu_count()
                pool = multiprocessing.Pool(ncpu)
                m = Vhat.shape[0]
                n = Vhat.shape[1]
                # set up argument for parallel programming
                args = zip(repeat(V,ncpu), repeat(r,ncpu),
                           repeat(niter[name2],ncpu),
                           repeat(min_error[name2], ncpu))
                result = pool.starmap(algo, args)
                pool.close()
                pool.join()
                errors_n = [i[2][-1] for i in result]
                W, H, errors = result[errors_n.index(min(errors_n))]
            else:
                W, H, errors = algo(V, r, niter[name2], min_error[name2])
            # save subVhat, V and H to disk - only epoch1
            if i == 0:
                path_matrix = os.path.join(folder, "matrix_{}".format(name))
                print("Saving to {}.npz".format(path_matrix))
                np.save(path_matrix, (subVhat, V, V_noise,  W.dot(H)))
            # plot error versus iteration
            data_name = folder.split("-")[-1]
            plotname = "{}_{}_Error_{}_Iteration".format(data_name, name2,
                                                         niter[name2])
            path = os.path.join(folder, plotname)
            error_vs_iter(errors, niter[name2], name2, path)
            # save errors to disk as well
            error_df = pd.DataFrame({"Errors": errors})
            error_df.to_csv("{}.csv".format(path), index=False)
            Ypred = util.assign_cluster_label(H.T, subYhat)

            # evaluate metrics
            _rre = metric.eval_rre(subVhat, W, H)
            _acc = metric.eval_acc(subYhat, Ypred)
            _nmi = metric.eval_nmi(subYhat, Ypred)
            print("Epoch = {}, Model = {}, RRE = {}".format(i + 1, name, _rre))
            print("Epoch = {}, Model = {}, ACC = {}".format(i + 1, name, _acc))
            print("Epoch = {}, Model = {}, NMI = {}".format(i + 1, name, _nmi))
            metrics["rre"][name].append(_rre)
            metrics["acc"][name].append(_acc)
            metrics["nmi"][name].append(_nmi)
    return metrics


def train(data_name, folder):
    """Run NMF on data stored in data_name."""
    # load dataset
    print("==> Load {} dataset...".format(data_name))
    Vhat, Yhat = io.load_data(data_name, scale[data_name.split(os.sep)[-1]])
    n = len(Yhat)
    size = int(n * sample_size)
    empty_metric = make_metrics()
    t = time.time()
    result = []
    for i in range(epoch):
        empty_metric = make_metrics()
        temp = one_simulation(i, Vhat, Yhat, n, size, empty_metric, folder)
        result.append(temp)
    t = time.time() - t
    print('done with time {} second'.format(t))

    # merge results together
    metrics = make_metrics()
    for dic in result:
        for metric_nm in dic:
            for model_nm in dic[metric_nm]:
                metrics[metric_nm][model_nm] += dic[metric_nm][model_nm]

    # get mean of metrics over epochs and save to disk
    mean_metrics = make_metrics()
    for metric_nm in metrics:
        for model_nm in metrics[metric_nm]:
            average = np.mean(metrics[metric_nm][model_nm])
            mean_metrics[metric_nm][model_nm] = average
    average_df = pd.DataFrame.from_dict(mean_metrics)
    average_df.to_csv(os.path.join(folder, 'statistics_large.csv'))
    print("Average of metrics: ")
    print(average_df)

    # split raw results and save to disk
    for metric_nm in metrics:
        filename = os.path.join(folder, 'raw_result_{}.csv'.format(metric_nm))
        raw = metrics[metric_nm]
        raw_df = pd.DataFrame.from_dict(raw).T
        raw_df = raw_df.reset_index()
        print("Saving to", filename)
        raw_df.to_csv(filename, header=False, index=False)

    # draw metrics comparison and save to folder
    for metric_nm in metrics:
        fig = pl.figure(figsize=(15, 6))
        ax = pl.subplot(111)
        for model_nm in metrics[metric_nm]:
            scores = metrics[metric_nm][model_nm]
            ax.plot(range(epoch), scores, label=model_nm)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_nm)
        ax.set_title("Model comparison of {}".format(metric_nm))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plot_name = os.path.join(folder, "metrics_{}".format(metric_nm))
        print("Saving to {}".format(plot_name))
        pl.savefig(plot_name+'.pdf')
        pl.savefig(plot_name+'.eps')


def make_metrics():
    """Make an empty metrics dictionary."""
    metrics = {"rre": {}, "acc": {}, "nmi": {}}
    for noise_fun in Noise:
        for name in model:
            name=name+' '+noise_fun
            metrics["rre"][name] = []
            metrics["acc"][name] = []
            metrics["nmi"][name] = []
    return metrics


if __name__ == "__main__":
    main()
