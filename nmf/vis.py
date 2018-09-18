"""Draw analysis plots."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from nmf import main, io

cmap = pl.cm.Greys
sample_index = 100


def draw_error(path1, path2):
    """Draw training error of two algorithms together."""
    pl.figure(figsize=(15, 6))
    for i, path in enumerate([path1, path2]):
        error = pd.read_csv(path)
        niter = int(path.split("_")[-2])
        dataname = path.split(os.sep)[-1].split("_")[0]
        algo_name = path.split(os.sep)[-1].split("_Error_")[0].split("_")[-1]
        x = min(len(error), niter)
        pl.subplot(1, 2, i + 1)
        pl.plot(np.arange(x), np.log(error))
        newticks = np.around(np.exp(pl.yticks()[0]), decimals=2)
        pl.yticks(pl.yticks()[0], newticks)
        pl.xlabel("Iteration")
        pl.ylabel("Training Error (Log Scale)")
        pl.title("{} {} Error versus {} Iteration"
                 .format(dataname, algo_name, niter))
    # pl.show()
    pl.savefig("Error.pdf", bbox_inches="tight")


def draw_noise(dataname):
    """Draw image before and after adding noise."""
    # load data
    assert dataname in ["ORL", "CroppedYaleB"]
    if dataname == "ORL":
        path = os.path.join("data", "ORL")
    else:
        path = os.path.join("data", "CroppedYaleB")
    print("==> Load {} dataset...".format(dataname))
    Vhat, Yhat = io.load_data(path, main.scale[dataname])
    # loop through noise
    my_scale = main.scale[dataname]
    my_img_size = main.img_size[dataname]
    for noise_name in main.Noise:
        if noise_name == "No Noise":
            continue
        noise_path = "Noise_{}_{}_Comparison.pdf".format(dataname, noise_name)
        V, V_noise = main.Noise[noise_name](Vhat)
        img_size = [x // my_scale for x in my_img_size]
        reshape_size = [img_size[1], img_size[0]]
        pl.figure(figsize=(10,6))
        pl.subplot(131)
        pl.imshow(Vhat[:, sample_index].reshape(reshape_size), cmap=cmap)
        pl.title('Image (Original)')
        pl.subplot(132)
        pl.imshow(V_noise[:, sample_index].reshape(reshape_size), cmap=cmap)
        pl.title("{} Noise".format(noise_name))
        pl.subplot(133)
        pl.imshow(V[:, sample_index].reshape(reshape_size), cmap=cmap)
        pl.title('Image (with {} Noise)'.format(noise_name))
        print("Saving to", noise_path)
        pl.savefig(noise_path, bbox_inches="tight")


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        print("Select from following option: error")
        sys.exit()
    if argv[1] == "error":
        path1, path2 = argv[2], argv[3]
        draw_error(path1, path2)
    elif argv[1] == "noise":
        draw_noise(argv[2])
