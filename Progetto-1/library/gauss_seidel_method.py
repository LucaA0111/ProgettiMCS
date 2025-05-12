import numpy as np

def gauss_seidel(A, b, tol=1e-6, max_iter=20000):
    x = np.zeros_like(b)
    n = len(b)
    for k in range(1, max_iter + 1):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(A @ x_new - b) / np.linalg.norm(b) < tol:
            return x_new, k
        x = x_new
    return x, k
