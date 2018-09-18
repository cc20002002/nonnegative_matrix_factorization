"""Noise module."""
import numpy as np


def normal(subVhat):
    """Design a Gaussian noise."""
    V_noise = np.random.normal(0, 80, subVhat.shape) #* np.sqrt(subVhat)
    V = subVhat + V_noise
    V[V < 0] = 0
    return V, V_noise


def possion(subVhat):
    """Design a Possion noise."""
    V = np.random.poisson(subVhat)
    V_noise = V-subVhat
    return V, V_noise


def salt_and_pepper(subVhat):
    """Design a salt and pepper noise where make some pixel value zeros."""
    # obtain one Image
    image = subVhat[:, 0]
    V_noise = np.random.randint(low=0, high=255, size=subVhat.shape, dtype=int)
    V = subVhat.copy()
    V[V_noise <= 20] = 0
    V[V_noise >= 230] = 255
    return V, V_noise


def identity(subVhat):
    """Add no noise to image."""
    V_noise = np.zeros(subVhat.shape)
    return subVhat, V_noise
