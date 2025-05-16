import numpy as np
from dctFunction import dct2_manual
from dctFunction import dct2_fast
from dctFunction import dct1_manual

def verify_dct2():
    f_mat = np.array([
        [231, 32, 233, 161, 24, 71, 140, 245],
        [247, 40, 248, 245, 124, 204, 36, 107],
        [234, 202, 245, 167, 9, 217, 239, 173],
        [193, 190, 100, 167, 43, 180, 8, 70],
        [11, 24, 210, 177, 81, 243, 8, 112],
        [97, 195, 203, 47, 125, 114, 165, 181],
        [193, 70, 174, 167, 41, 30, 127, 245],
        [87, 149, 57, 192, 65, 129, 178, 228]
    ])

    expected_mat = np.array([
        [1.11e+03, 4.40e+01, 7.59e+01, -1.38e+02, 3.50e+00, 1.22e+02, 1.95e+02, -1.01e+02],
        [7.71e+01, 1.14e+02, -2.18e+01, 4.13e+01, 8.77e+00, 9.90e+01, 1.38e+02, 1.09e+01],
        [4.48e+01, -6.27e+01, 1.11e+02, -7.63e+01, 1.24e+02, 9.55e+01, -3.98e+01, 5.85e+01],
        [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01, 2.66e+01, -3.68e+01, 6.61e+01, 1.25e+02],
        [-1.09e+02, -4.33e+01, -5.55e+01, 8.17e+00, 3.02e+01, -2.86e+01, 2.44e+00, -9.41e+01],
        [-5.38e+00, 5.66e+01, 1.73e+02, -3.54e+01, 3.23e+01, 3.34e+01, -5.81e+01, 1.90e+01],
        [7.88e+01, -6.45e+01, 1.18e+02, -1.50e+01, -1.37e+02, -3.06e+01, -1.05e+02, 3.98e+01],
        [1.97e+01, -7.81e+01, 9.72e-01, -7.23e+01, -2.15e+01, 8.13e+01, 6.37e+01, 5.90e+00]
    ])

    tolerance = 1e-2
    # Calcolo della DCT2 manuale
    c_mat_manual = dct2_manual(f_mat)

    # Differenza tra la matrice calcolata e quella attesa
    difference = np.linalg.norm(c_mat_manual - expected_mat)

    print("\n=== Verifica DCT2 ===")
    print("Risultati ottenuti:")
    print(str(c_mat_manual))
    print("\nRisultati attesi:")
    print(str(expected_mat))
    print(f"\nDifferenza: {difference:.6e}\n")

    # Risultato del confronto
    if difference < tolerance:
        print(f"Successo: La DCT2 manuale coincide con i risultati attesi (differenza = {difference:.6e}).")
        return True
    else:
        print(f"Errore: La DCT2 manuale NON coincide con i risultati attesi (differenza = {difference:.6e}).")
        return False

def verify_dct1():

    row = np.array([231, 32, 233, 161, 24, 71, 140, 245])

    #expected_row = np.array([4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02, 6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01])
    expected_row = np.array([401.00, 6.60, 109.00, -112.00, 65.40, 121.00, 116.00, 28.80])

    tolerance = 1e-2

    dct1_result = dct1_manual(row)
    difference = np.linalg.norm(dct1_result - expected_row)

    print("\n=== Verifica DCT1 ===")
    print("Risultato ottenuto:")
    print(dct1_result)
    print("\nRisultato atteso:")
    print(expected_row)
    print(f"\nDifferenza: {difference:.6e}\n")

    if difference < tolerance:
        print(f"Successo: La DCT1 manuale coincide con i risultati attesi (differenza = {difference:.6e}).")
        return True
    else:
        print(f"Errore: La DCT1 manuale NON coincide con i risultati attesi (differenza = {difference:.6e}).")
        return False