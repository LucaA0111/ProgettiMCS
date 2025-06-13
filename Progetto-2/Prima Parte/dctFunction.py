import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fft import dct, idct


class DCT2:
    """
        Classe per il calcolo della Trasformata Discreta di Coseno 2D (DCT2),
        con implementazione manuale e veloce (usando SciPy).

        Metodi:
            - compute_D(N): Calcola la matrice D della DCT di dimensione NxN.
            - dct2_manual(f_mat): Calcola la DCT2 di una matrice NxN manualmente usando la matrice D.
            - dct2_fast(f_mat): Calcola la DCT2 di una matrice NxN usando la funzione ottimizzata di SciPy.
            - verify_dct(): Esegue test di verifica confrontando i risultati manuali e SciPy con valori attesi.
    """

    def __init__(self):
        """
            Inizializza la classe DCT2. Non richiede parametri.
        """
        pass

    def compute_D(self, N):
        """
            Calcola la matrice di trasformazione D della DCT 1D di dimensione N x N.

            Args:
                N (int): Dimensione della matrice quadrata.

            Returns:
                numpy.ndarray: Matrice NxN della trasformata DCT 1D con normalizzazione ortogonale.

            Dettagli:
                La matrice D è definita con elementi:
                D[k,j] = coef * cos(pi * k * (2j + 1) / (2N))
                dove coef = sqrt(1/N) per k=0, altrimenti sqrt(2/N).
        """

        D = np.zeros((N, N))  # inizializza matrice D a zeri
        for k in range(N):
            if k == 0:
                coef = np.sqrt(1 / N)  # coefficiente di normalizzazione per k = 0
            else:
                coef = np.sqrt(2 / N)  # coefficiente di normalizzazione per k > 0

            for j in range(N):
                # Calcolo dell'elemento (k, j) della matrice D secondo la formula della DCT
                D[k, j] = coef * np.cos((np.pi * k * (2 * j + 1)) / (2 * N))

        return D

    def dct2_manual(self, f_mat):
        """
            Calcola la Trasformata Discreta di Coseno 2D di una matrice NxN
            usando la matrice D calcolata manualmente.

            Args:
                f_mat (numpy.ndarray): Matrice di input quadrata NxN (ad esempio immagine o blocco di dati).

            Returns:
                numpy.ndarray: Matrice NxN contenente i coefficienti DCT2.

            Metodo:
                - Applica la DCT 1D (moltiplicazione per D) lungo le colonne.
                - Applica la DCT 1D lungo le righe (moltiplicazione per D^T).
        """

        N = f_mat.shape[0]
        D = self.compute_D(N)  # calcola la matrice D

        # Copia della matrice di input
        c_mat = f_mat.copy()

        # Applica la DCT alle colonne: c = D @ f
        for j in range(N):
            c_mat[:, j] = D @ c_mat[:, j]  # moltiplicazione matrice-vettore colonna per ogni colonna

        # Applica la DCT alle righe: c = c @ D^T
        for i in range(N):
            c_mat[i, :] = (D @ c_mat[i, :].T).T  # moltiplicazione matrice-vettore riga

        return c_mat

    def dct2_fast(self, f_mat):

        """
            Calcola la Trasformata Discreta di Coseno 2D utilizzando la funzione ottimizzata
            dct di SciPy (trasformata separabile su righe e colonne).

            Args:
                f_mat (numpy.ndarray): Matrice di input quadrata NxN.

            Returns:
                numpy.ndarray: Matrice NxN contenente i coefficienti DCT2.

            Nota:
                Utilizza la normalizzazione ortogonale ('ortho') per garantire coerenza.
        """

        # Applica DCT lungo l’asse 0 (righe), poi lungo l’asse 1 (colonne), con normalizzazione ortogonale
        return dct(dct(f_mat, axis=0, norm='ortho'), axis=1, norm='ortho')

    def verify_dct(self):
        """
            Esegue una verifica della correttezza delle implementazioni DCT1D e DCT2.

            Dettagli:
                - Usa un blocco di test 8x8 con valori noti.
                - Confronta la DCT 1D calcolata manualmente sulla prima riga con valori attesi.
                - Confronta la DCT2 calcolata manualmente con quella con i valori attesi
                - Stampa i risultati e le differenze assolute per analisi.

            Nota:
                Le differenze nei risultati possono essere dovute a diverse convenzioni di normalizzazione.
        """

        # Blocco di test 8x8 fornito nel progetto
        test_block = np.array([
            [231, 32, 233, 161, 24, 71, 140, 245],
            [247, 40, 248, 245, 124, 204, 36, 107],
            [234, 202, 245, 167, 9, 217, 239, 173],
            [193, 190, 100, 167, 43, 180, 8, 70],
            [11, 24, 210, 177, 81, 243, 8, 112],
            [97, 195, 203, 47, 125, 114, 165, 181],
            [193, 70, 174, 167, 41, 30, 127, 245],
            [87, 149, 57, 192, 65, 129, 178, 228]
        ])

        expected_dct2 = np.array([
            [1.11e+03, 4.40e+01, 7.59e+01, -1.38e+02, 3.50e+00, 1.22e+02, 1.95e+02, -1.01e+02],
            [7.71e+01, 1.14e+02, -2.18e+01, 4.13e+01, 8.77e+00, 9.90e+01, 1.38e+02, 1.09e+01],
            [4.48e+01, -6.27e+01, 1.11e+02, -7.63e+01, 1.24e+02, 9.55e+01, -3.98e+01, 5.85e+01],
            [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01, 2.66e+01, -3.68e+01, 6.61e+01, 1.25e+02],
            [-1.09e+02, -4.33e+01, -5.55e+01, 8.17e+00, 3.02e+01, -2.86e+01, 2.44e+00, -9.41e+01],
            [-5.38e+00, 5.66e+01, 1.73e+02, -3.54e+01, 3.23e+01, 3.34e+01, -5.81e+01, 1.90e+01],
            [7.88e+01, -6.45e+01, 1.18e+02, -1.50e+01, -1.37e+02, -3.06e+01, -1.05e+02, 3.98e+01],
            [1.97e+01, -7.81e+01, 9.72e-01, -7.23e+01, -2.15e+01, 8.13e+01, 6.37e+01, 5.90e+00]

        ])

        # Prima riga per il test 1D
        first_row = test_block[0, :]

        # Applica la DCT1D alla prima riga con la matrice D
        D = self.compute_D(8)
        dct_row = D @ first_row  # moltiplicazione matrice D per vettore riga

        expected_dct_row = np.array([4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02,6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01])

        # Esegue DCT2 manuale sul blocco di test
        dct2_result = self.dct2_manual(test_block)
        # Esegue DCT2 veloce di SciPy sul blocco di test
        dct2_scipy = self.dct2_fast(test_block)


        print("\n➡VERIFICA DCT 1D:\n")
        print("Risultato ottenuto:")
        print(dct_row)
        print("\nRisultato atteso:")
        print(expected_dct_row)
        print("\nDifferenza assoluta:")
        print(np.abs(dct_row - expected_dct_row))
        print("------")

        print("\n➡VERIFICA DCT2:\n")
        print("DCT2 manuale:\n")
        print(dct2_result)
        print("\nDCT2 con SciPy:")
        print(dct2_scipy)
        print("\nDifferenza assoluta:")
        print(np.abs(dct2_result - expected_dct2))