import numpy as np

def conjugate_gradient(A, b, tol=1e-6, max_iter=20000):
    x = np.zeros_like(b)
    r = b - A @ x
    p = np.copy(r)
    rs_old = np.dot(r, r)
    for k in range(1, max_iter + 1):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) / np.linalg.norm(b) < tol:
            return x, k
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, k
