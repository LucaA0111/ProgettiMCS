import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_results(data):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("plots", now)
    os.makedirs(base_dir, exist_ok=True)

    metrics = [
        ('iters', 'linear', 'Numero di Iterazioni'),
        ('error', 'log', 'Errore Relativo'),
        ('times', 'log', 'Tempo di Esecuzione (s)')
    ]

    for matrix in data['matrices']:
        results = data['results'][matrix]
        matrix_name = os.path.splitext(matrix)[0]
        matrix_dir = os.path.join(base_dir, matrix_name)
        os.makedirs(matrix_dir, exist_ok=True)

        for metric, yscale, ylabel in metrics:
            plt.figure(figsize=(8, 6))
            for solver in data['solvers']:
                if metric == 'times':
                    y = [
                        results[tol][metric][solver]
                        for tol in data['tols']
                    ]
                else:
                    metric_idx = 0 if metric == 'iters' else 1
                    y = [
                        results[tol]['results'][solver][metric_idx]
                        for tol in data['tols']
                    ]

                plt.plot(data['tols'], y, marker='o', label=solver)

            plt.xscale('log')
            plt.yscale(yscale)
            plt.xlabel('Tolleranza')
            plt.ylabel(ylabel)
            plt.title(f"{matrix_name} - {ylabel}")
            plt.legend()
            plt.grid(True, which='both', ls='--', lw=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(matrix_dir, f"{metric}.png"))
            plt.show()
            plt.close()
