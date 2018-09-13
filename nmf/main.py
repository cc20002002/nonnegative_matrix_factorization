import numpy as np
import os
import pandas as pd
from nmf import io, util, metric, algorithm
import matplotlib.pyplot as pl
#from functools import partial
import multiprocessing 
from itertools import repeat
import time
from numba import jit
from numba import vectorize, float64, int64
# Configuration
sample_index = 100
sample_size = 0.9
epoch = 2
cmap = pl.cm.Greys
random_state = 0
reduce_scale_yaleB = 4
reduce_scale_orl = 3
orl_img_size = (92, 112)
yaleB_img_size = (168, 192)
parallel_flag=1
niter = {
    #"Benchmark (scikit-learn)": algorithm.benchmark,
    "Multiplication KL Divergence": 5000,
    "Multiplication Euclidean": 5000,
    # "Truncated Cauchy": algorithm.truncated_cauchy,
}

min_error = {
    #"Benchmark (scikit-learn)": algorithm.benchmark,
    "Multiplication KL Divergence": 2.325,
    "Multiplication Euclidean": 470,
    # "Truncated Cauchy": algorithm.truncated_cauchy,
}
model = {
    #"Benchmark (scikit-learn)": algorithm.benchmark,
    "Multiplication KL Divergence": algorithm.multiplication_divergence,
    "Multiplication Euclidean": algorithm.multiplication_euclidean,
    # "Truncated Cauchy": algorithm.truncated_cauchy,
}

Noise = ["Poisson","Normal"]

 

def main():
    """Run NMF on CroppedYaleB and ORL dataset."""
    if os.name == 'nt123':
        train("..\\data\\ORL")
        # train("data/CroppedYaleB")
    else:
        train("data/ORL")
        # train("data/CroppedYaleB")

def one_simulation(i,Vhat,Yhat,n,size,metrics):
    print("Epoch {}...".format(i + 1))
    # sample 90% of samples
    index = np.random.choice(np.arange(n), size, replace=False)
    subVhat, subYhat = Vhat[:, index], Yhat[index]
    for noise_fun in Noise:
        if noise_fun=='Normal':
            V_noise = np.random.normal(0, 5, subVhat.shape) #* np.sqrt(subVhat)
            V=subVhat+V_noise
            V[V<=0]=1e-12
        elif noise_fun=='Poisson':
            V = np.random.poisson(subVhat)
            V_noise = V-subVhat
        
            # if i == 0:
            #     draw_image(V, subVhat, V_noise, sample_index)

        r = np.unique(Yhat).shape[0]
        # loop through different models
        for name, algo in model.items():
            name2=name
            name=name+' '+noise_fun               
            print(name)
            W, H = algo(V, r,niter[name2],min_error[name2])
            Ypred = util.assign_cluster_label(H.T, subYhat)

            # evaluate metrics
            _rre = metric.eval_rre(subVhat, W, H)
            _acc = metric.eval_acc(subYhat, Ypred)
            _nmi = metric.eval_nmi(subYhat, Ypred)
            print("RRE = {}".format(_rre))
            print("ACC = {}".format(_acc))
            print("NMI = {}".format(_nmi))

            metrics["rre"][name].append(_rre)
            metrics["acc"][name].append(_acc)
            metrics["nmi"][name].append(_nmi)
    return metrics

    # performace over epochs


def train(data_name):
    """Run NMF on data stored in data_name."""
    # load ORL dataset
    print("==> Load {} dataset...".format(data_name))
    Vhat, Yhat = io.load_data(data_name, reduce_scale_orl)

    n = len(Yhat)
    size = int(n * sample_size)
    metrics = {"rre": {}, "acc": {}, "nmi": {}}
    for noise_fun in Noise:
        for name in model:
            name=name+' '+noise_fun
            metrics["rre"][name] = []
            metrics["acc"][name] = []
            metrics["nmi"][name] = []
    t = time.time()
    if parallel_flag:
        pool = multiprocessing.Pool(os.cpu_count())
        #sim=partial(one_simulation,Vhat=Vhat,Yhat=Yhat,n=n,size=size,metrics=metrics)
        
        #pool.starmap(sim, (range(epoch),))
        metrics=pool.starmap(one_simulation,zip(range(epoch),repeat(Vhat),repeat(Yhat),repeat(n),repeat(size),repeat(metrics)))
        pool.close()
        pool.join()
    else:
        result=[]
        for i in range(epoch):
            temp=one_simulation(i,Vhat,Yhat,n,size,metrics)
            result.append(temp)
        metrics=result          
    t = time.time()-t   
    print('done')
    mean_metrics = {}
    for mname in ["rre", "acc", "nmi"]:
        mean_metrics[mname] = {}
        for name in model:
            for noise_fun in Noise:
                mean_metrics[mname][name+' '+noise_fun]=0
                for i in range(epoch):
                    mean_metrics[mname][name+' '+noise_fun] += metrics[i][mname][name+' '+noise_fun][0]
                mean_metrics[mname][name+' '+noise_fun]=mean_metrics[mname][name+' '+noise_fun]/epoch
    df = pd.DataFrame.from_dict(mean_metrics)
    print(df)
    df.to_csv('statistics_large.csv')
    for mname in ["rre", "acc", "nmi"]:
        if parallel_flag:
            for i in (range(0,epoch,3)):
                if i==0 & (mname=="rre"):
                    raw_result = pd.DataFrame.from_dict(metrics[i][mname])
                    raw_result.to_csv('raw_result_large_'+mname+'.csv')
                else:
                    raw_result = pd.DataFrame.from_dict(metrics[i][mname])
                    raw_result.to_csv('raw_result_large_'+mname+'.csv', mode='a', header=False)
        else:
            i=0
            if mname=="rre":
                raw_result = pd.DataFrame.from_dict(metrics[i][mname])
                raw_result.to_csv('raw_result_large_'+mname+'.csv')
            else:
                raw_result = pd.DataFrame.from_dict(metrics[i][mname])
                raw_result.to_csv('raw_result_large_'+mname+'.csv', mode='a', header=False)
    import IPython; IPython.embed() 
    for name in model:
        for noise_fun in Noise:
            rres = metrics["rre"][name+' '+noise_fun]
            pl.plot(range(epoch), np.log(rres), label=name+' '+noise_fun)
    pl.legend(loc="lower right")
    pl.xlabel("epoch")
    pl.ylabel("relative reconstruction error")
    pl.title("Model comparison of RRE")
    pl.show()    







def draw_image(V, subVhat, V_noise, sample_index):
    """Draw image before and after adding noise."""
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

if __name__ == "__main__":
    main()

