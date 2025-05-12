import numpy as np

def gradient(A, b, tol=1e-6, max_iter=20000):
    x = np.zeros_like(b)
    r = b - A @ x
    for k in range(1, max_iter + 1):
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        if np.linalg.norm(r) / np.linalg.norm(b) < tol:
            return x, k
    return x, k
