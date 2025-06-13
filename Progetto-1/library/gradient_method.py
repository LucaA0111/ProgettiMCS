import numpy as np

def gradient(A, b, tol=1e-6, max_iter=20000):
    """
        Risolve il sistema lineare Ax = b usando il metodo del gradiente.

        Metodo iterativo che minimizza la funzione quadratica associata al sistema,
        aggiornando la soluzione lungo la direzione del gradiente residuo.

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
    r = b - A @ x
    for k in range(1, max_iter + 1):
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        if np.linalg.norm(r) / np.linalg.norm(b) < tol:
            return x, k
    return x, k
