
def jacobi(A, b, tol=1e-6, max_iter=20000):
    """
        Risolve il sistema lineare Ax = b usando il metodo iterativo di Jacobi.

        Il metodo aggiorna iterativamente la soluzione basandosi sulla
        decomposizione della matrice A in parte diagonale e resto.

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
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        if np.linalg.norm(A @ x_new - b) / np.linalg.norm(b) < tol:
            return x_new, k
        x = x_new
    return x, k
