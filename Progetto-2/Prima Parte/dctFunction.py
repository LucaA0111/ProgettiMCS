import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fftpack import dct

#TODO: arrotondamenti dct2 e dct1
def compute_D(N):
    alpha_vect = np.zeros(N)
    alpha_vect[0] = 1 / np.sqrt(N)
    alpha_vect[1:] = np.sqrt(2 / N)

    D = np.zeros((N, N))
    for k in range(N):
        for i in range(N):
            D[k, i] = alpha_vect[k] * np.cos((k * np.pi * (2 * i + 1)) / (2 * N))

    return D


def dct1_manual(row):
    N = len(row)
    D = compute_D(N)
    return np.dot(D, row)

# Funzione per calcolare la DCT2 manuale
def dct2_manual(f_mat):
    N = f_mat.shape[0]
    # Calcolo della matrice DCT
    D = compute_D(N)

    # Creazione della matrice di risultato
    c_mat = np.copy(f_mat)

    # Fase 1: DCT lungo le colonne
    c_mat = np.dot(D, c_mat)

    # Fase 2: DCT lungo le righe
    c_mat = np.dot(D, c_mat.T).T


    return c_mat

# Funzione per calcolare la DCT2 usando scipy (DCT veloce)
def dct2_fast(f_mat):
    # DCT lungo le colonne (axis=0)
    temp = dct(f_mat, type=2, norm='ortho', axis=0)

    # DCT lungo le righe (axis=1)
    c_mat = dct(temp, type=2, norm='ortho', axis=1)

    return c_mat

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

