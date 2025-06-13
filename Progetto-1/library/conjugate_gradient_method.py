import numpy as np

def conjugate_gradient(A, b, tol=1e-6, max_iter=20000):
    """
        Risolve il sistema lineare Ax = b usando il metodo del gradiente coniugato.

        Metodo iterativo che minimizza la funzione quadratica associata al sistema,
        utilizzando direzioni di ricerca coniugate per garantire convergenza più rapida
        rispetto al metodo del gradiente semplice.

        Parameters:
            A: ndarray
                Matrice quadrata dei coefficienti, simmetrica e definita positiva.
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
