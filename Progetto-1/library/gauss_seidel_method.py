import numpy as np
from lower_triangular_method import lower_triangular


def gauss_seidel(A, b, tol=1e-6, max_iter=20000):
    """
        Risolve il sistema lineare Ax = b con il metodo iterativo di Gauss-Seidel.

        Il metodo utilizza la decomposizione di A in parte triangolare inferiore (L)
        e parte strettamente superiore (U), aggiornando iterativamente la soluzione
        risolvendo sistemi triangolari inferiori.

        Parameters:
            A: ndarray
                Matrice quadrata dei coefficienti.
            b: ndarray
                Vettore dei termini noti, di lunghezza n.
            tol: float, optional
                Tolleranza relativa per il criterio di arresto. Default è 1e-6.
            max_iter: int, optional
                Numero massimo di iterazioni. Default è 20000.

        Returns:
            x: ndarray
                Vettore soluzione approssimata di Ax = b.
            k: int
                Numero di iterazioni eseguite.
        """

    x = np.zeros_like(b)
    n = len(b)

    # Estrae la parte triangolare inferiore L (inclusa la diagonale)
    L = np.tril(A)
    # Calcola la parte strettamente superiore U
    U = A - L

    for k in range(1, max_iter + 1):
        # Calcolo del termine noto per il sistema triangolare inferiore
        rhs = b - np.dot(U, x)
        # Risolve il sistema triangolare inferiore
        x_new = lower_triangular(L, rhs)

        # Criterio di arresto
        if np.linalg.norm(A @ x_new - b) / np.linalg.norm(b) < tol:
            return x_new, k
        x = x_new

    return x, k