import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import time

# Funzione per calcolare la DCT2 manuale
def dct2_manual(f_mat):
    N = f_mat.shape[0]
    D = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            D[k, n] = np.cos(np.pi * (n + 0.5) * k / N)

    # DCT1 per le colonne
    c_mat = f_mat.copy()
    for j in range(N):
        c_mat[:, j] = np.dot(D, c_mat[:, j])

    # DCT1 per le righe
    for j in range(N):
        c_mat[j, :] = np.dot(D, c_mat[j, :].T).T

    return c_mat


# Funzione per calcolare la DCT2 usando scipy (DCT veloce)
def dct2_fast(f_mat):
    return fft.dct(fft.dct(f_mat.T, type=2).T, type=2)

# Funzione per confrontare la DCT manuale e la DCT veloce
def compare_dct(n_values):
    manual_times = []
    fast_times = []

    for N in n_values:
        # Crea una matrice f_mat casuale NxN
        f_mat = np.random.rand(N, N)

        # Misura il tempo per la DCT manuale
        start_time = time.time()
        dct_manual = dct2_manual(f_mat)
        manual_time = time.time() - start_time
        manual_times.append(manual_time)

        # Misura il tempo per la DCT veloce
        start_time = time.time()
        dct_fast = dct2_fast(f_mat)
        fast_time = time.time() - start_time
        fast_times.append(fast_time)

    # Grafico dei tempi in scala semilogaritmica
    plt.figure(figsize=(8, 6))
    plt.plot(n_values, manual_times, marker='o', label='DCT Manuale', color='b')
    plt.plot(n_values, fast_times, marker='x', label='DCT Veloce', color='r')
    plt.xlabel('Dimensione della matrice (N)')
    plt.ylabel('Tempo di esecuzione (secondi)')
    plt.title('Confronto tra DCT Manuale e DCT Veloce')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

