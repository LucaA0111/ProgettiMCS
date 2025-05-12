import time
import numpy as np
from scipy.io import mmread
from library.jacobi_method import jacobi
from library.gauss_seidel_method import gauss_seidel
from library.gradient_method import gradient
from library.conjugate_gradient_method import conjugate_gradient
from library.plot import plot_results

matrix_files = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
tols = [1e-4, 1e-6, 1e-8, 1e-10]
solvers = {
    "Jacobi": jacobi,
    "Gauss-Seidel": gauss_seidel,
    "Gradient": gradient,
    "Conjugate Gradient": conjugate_gradient
}

print("Seleziona le matrici (digita 0 per utilizzarle tutte):")
for i, mat in enumerate(matrix_files, 1):
    print(f"{i}. {mat}")
matrix_choice = list(map(int, input("Inserisci i numeri separati da spazio: ").split()))
if 0 in matrix_choice:
    selected_matrices = matrix_files
else:
    selected_matrices = [matrix_files[i - 1] for i in matrix_choice if 1 <= i <= len(matrix_files)]

print("\nSeleziona le tolleranze (digita 0 per utilizzarle tutte):")
for i, tol in enumerate(tols, 1):
    print(f"{i}. {tol}")
tol_choice = list(map(int, input("Inserisci i numeri separati da spazio: ").split()))
if 0 in tol_choice:
    selected_tols = tols
else:
    selected_tols = [tols[i - 1] for i in tol_choice if 1 <= i <= len(tols)]

all_results = {}

for file in selected_matrices:
    print(f"\n==== Matrix: {file} ====")
    A = mmread(f"dati/{file}").toarray()
    x_exact = np.ones(A.shape[0])
    b = A @ x_exact
    matrix_results = {}

    for tol in selected_tols:
        print(f"\nTOL = {tol}")
        method_results = {}
        time_results = {}
        for name, method in solvers.items():
            start = time.time()
            x_approx, iters = method(A, b, tol=tol)
            elapsed = time.time() - start
            error = np.linalg.norm(x_exact - x_approx) / np.linalg.norm(x_exact)
            print(f"{name:20} | iter: {iters:5d} | error: {error:.2e} | time: {elapsed:.4f}s")
            method_results[name] = (iters, error, elapsed)
            time_results[name] = elapsed
        matrix_results[tol] = {'results': method_results, 'times': time_results}
    all_results[file] = matrix_results

data = {
    'results': all_results,
    'tols': selected_tols,
    'solvers': list(solvers.keys()),
    'matrices': selected_matrices
}

plot_results(data)
