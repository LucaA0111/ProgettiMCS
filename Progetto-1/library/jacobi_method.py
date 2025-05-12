import numpy as np

def jacobi(A, b, tol=1e-6, max_iter=20000):
    x = np.zeros_like(b)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        if np.linalg.norm(A @ x_new - b) / np.linalg.norm(b) < tol:
            return x_new, k
        x = x_new
    return x, k
