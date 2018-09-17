"""Noise module."""
import numpy as np


def normal(subVhat):
    """Design a Gaussian noise."""
    V_noise = np.random.normal(0, 20, subVhat.shape) #* np.sqrt(subVhat)
    V = subVhat + V_noise
    V[V < 0] = 0
    return V, V_noise


def possion(subVhat):
    """Design a Possion noise."""
    V = np.random.poisson(subVhat)
    V_noise = V-subVhat
    return V, V_noise


def random(subVhat):
    """Design a random noise where make some pixel value zeros."""
    size = 0.3
    nrow, ncol = subVhat.shape
    row_index = np.random.choice(nrow, int(nrow * size))
    col_index = np.random.choice(ncol, int(ncol * size))
    V = subVhat.copy()
    for i, j in zip(row_index, col_index):
        V[i, j] = 0
    V_noise = V - subVhat
    return V, V_noise
