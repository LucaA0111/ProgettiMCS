import numpy as np


def lower_triangular(L, b):
    """
    Risolve il sistema lineare Lx = b con L triangolare inferiore.

    Utilizza la sostituzione in avanti. Si assume che L abbia zeri sopra la diagonale
    e che tutti gli elementi diagonali siano diversi da zero.

    Parameters:
        L : ndarray
            Matrice quadrata triangolare inferiore.
        b : ndarray
            Vettore dei termini noti, di lunghezza n.

    Returns:
        x : ndarray
            Vettore soluzione di Lx = b.
    """

    n = len(b)
    x = np.zeros_like(b)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x