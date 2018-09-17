"""Draw analysis plots."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


def draw_error(path1, path2):
    """Draw training error of two algorithms together."""
    pl.figure(figsize=(15, 6))
    for i, path in enumerate([path1, path2]):
        error = pd.read_csv(path)
        niter = int(path.split("_")[-2])
        dataname = path.split("/")[-1].split("_")[0]
        algo_name = path.split("/")[-1].split("_Error_")[0].split("_")[-1]
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
    pl.savefig("Error")


if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        print("Select from following option: error")
        sys.exit()
    if argv[1] == "error":
        path1, path2 = argv[2], argv[3]
        draw_error(path1, path2)
