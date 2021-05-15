from enum import Enum
import numpy as np


kernel_options = Enum("KernelOptions", "linear rbf polynomial")


def linear_kernel(X, Y=None):
    """
    Compute linear kernel pairwise for each x in X and y in Y

    Input:
        - X matrix of shape m x n   (i.e. m examples each of shape 1 x n)
        - Y matrix of shape p x n   (i.e. p examples each of shape 1 x n)
    Returns:
        - a matrix of shape m x p where each entry (m, p) stores the kernel value K(m, p)
    """

    if Y is None:
        return X @ X.T
    
    return X @ Y.T


def rbf_kernel(X, Y=None, sigma=1):
    """
    Compute rbf kernel pairwise for each x in X and y in Y

    Input:
        - X matrix of shape m x n   (i.e. m examples each of shape 1 x n)
        - Y matrix of shape p x n   (i.e. p examples each of shape 1 x n)
    Returns:
        - a matrix of shape m x p where each entry (m, p) stores the kernel value K(m, p)
    """

    if Y is None:
        Y = X

    X = X.reshape(X.shape[0], 1, X.shape[1])
    norm_sq = np.power(np.linalg.norm(X - Y, axis=2), 2)

    exp_pow = -1 * norm_sq / (2 * (sigma ** 2))

    return np.exp(exp_pow)


def polynomial_kernel():
    """
    TODO
    """

    pass