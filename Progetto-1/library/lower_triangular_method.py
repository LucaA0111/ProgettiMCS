import numpy as np


def lower_triangular(L, b):

    n = len(b)
    x = np.zeros_like(b)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x