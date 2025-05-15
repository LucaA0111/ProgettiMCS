import numpy as np
from lower_triangular_method import lower_triangular


def gauss_seidel(A, b, tol=1e-6, max_iter=20000):

    x = np.zeros_like(b)
    n = len(b)

    # Estrai la parte triangolare inferiore L (inclusa la diagonale)
    L = np.tril(A)
    # Calcola la parte strettamente superiore U
    U = A - L

    for k in range(1, max_iter + 1):
        # Calcolo del termine noto per il sistema triangolare inferiore
        rhs = b - np.dot(U, x)
        # Risolvi il sistema triangolare inferiore
        x_new = lower_triangular(L, rhs)

        # Criterio di arresto
        if np.linalg.norm(A @ x_new - b) / np.linalg.norm(b) < tol:
            return x_new, k
        x = x_new

    return x, k