import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from dctFunction import DCT2

"""
Modulo principale per il benchmarking delle implementazioni della trasformata DCT2.

Importa:
- numpy per la manipolazione matriciale,
- matplotlib per la visualizzazione dei dati,
- time per misurazioni temporali,
- os e datetime per gestione file e timestamp,
- DCT2 dalla libreria esterna dctFunction per le trasformate DCT2.

La classe Main esegue test di performance (benchmark) confrontando l'implementazione manuale
e quella veloce (SciPy), visualizza i risultati e salva i grafici in cartelle con timestamp.

Esecuzione:
Esegue una verifica dei risultati DCT2, quindi un benchmark per varie dimensioni di matrici,
infine mostra e salva grafici con i tempi di esecuzione e il rapporto di velocità.
"""

class Main:
    """
        Classe principale per gestire il benchmarking e la visualizzazione delle prestazioni
        delle implementazioni della trasformata discreta del coseno bidimensionale (DCT2).

        Attributi:
            transformer (DCT2): istanza della classe DCT2 per calcolare le trasformate.
            output_dir (str): percorso della cartella dove vengono salvati i grafici con timestamp.
    """

    def __init__(self):
        """
            Inizializza la classe Main creando un'istanza di DCT2 e la cartella di output
            per i risultati, nominata con il timestamp corrente (formato YYYYMMDD_HHMMSS).
        """
        self.transformer = DCT2()

        # Crea la cartella per salvare i risultati con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("plots", f"{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

    def benchmark(self, sizes):
        """
            Esegue il benchmark del calcolo della DCT2 su matrici quadrate di dimensioni specificate.

            Parametri:
                sizes (list of int): lista delle dimensioni N delle matrici NxN da testare.

            Ritorna:
                dict: dizionario con i seguenti campi:
                    - 'sizes': dimensioni testate,
                    - 'manual_times': tempi di esecuzione dell'implementazione manuale,
                    - 'fast_times': tempi di esecuzione dell'implementazione veloce (SciPy).
        """
        results = {
            'sizes': sizes,
            'manual_times': [],
            'fast_times': []
        }

        for N in sizes:
            print(f"-Matrice {N}x{N}")

            # Crea una matrice casuale NxN
            f_mat = np.random.rand(N, N)

            # Misura il tempo per l'implementazione manuale
            start_time = time.time()
            self.transformer.dct2_manual(f_mat)
            manual_time = time.time() - start_time
            results['manual_times'].append(manual_time)

            # Misura il tempo per l'implementazione veloce (più esecuzioni per dimensioni piccole)
            num_runs = 10 if N < 32 else 1
            fast_times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.transformer.dct2_fast(f_mat)
                fast_times.append(time.time() - start_time)

            # Calcola il tempo medio
            fast_time = sum(fast_times) / num_runs
            results['fast_times'].append(fast_time)

            print(f"  - Manuale: {manual_time:.6f} s \n  - Veloce: {fast_time:.6f} s")

        return results

    def plot_results(self, results):
        """
            Genera e salva grafici per visualizzare i risultati del benchmark.

            Parametri:
                results (dict): dizionario restituito dal metodo benchmark, con tempi e dimensioni.

            Funzionalità:
                - Grafico con scala logaritmica dei tempi di esecuzione vs dimensione,
                  includendo curve teoriche di complessità O(N³) e O(N² log N).
                - Grafico del rapporto di velocità (manuale / veloce).
                - Salvataggio delle immagini PNG nella cartella di output.
        """

        plt.figure(figsize=(10, 6))

        # Grafico: tempo vs dimensione della matrice
        plt.semilogy(results['sizes'], results['manual_times'], 'o-', label='DCT2 Manuale (O(N³))')
        plt.semilogy(results['sizes'], results['fast_times'], 's-', label='DCT2 Veloce (O(N²log(N)))')

        # Aggiunge curve teoriche di complessità come riferimento
        sizes_np = np.array(results['sizes'])

        # Scala le curve per farle combaciare con i dati misurati
        for i, val in enumerate(results['manual_times']):
            if val > 0:
                manual_scale = val / (sizes_np[i] ** 3)
                break
        else:
            manual_scale = 1e-10

        for i, val in enumerate(results['fast_times']):
            if val > 0:
                fast_scale = val / (sizes_np[i] ** 2 * np.log(sizes_np[i]))
                break
        else:
            fast_scale = 1e-10

        # Curve di riferimento
        plt.semilogy(sizes_np, manual_scale * sizes_np ** 3, '--', label='Riferimento O(N³)')
        plt.semilogy(sizes_np, fast_scale * sizes_np ** 2 * np.log(sizes_np), '--', label='Riferimento O(N²log(N))')

        plt.xlabel('Dimensione matrice (N)')
        plt.ylabel('Tempo di esecuzione (s)')
        plt.title('Confronto tempo di esecuzione DCT2')
        plt.legend()
        plt.grid(True)

        # Annotazioni con il rapporto dei tempi
        for i, N in enumerate(results['sizes']):
            manual_time = results['manual_times'][i]
            fast_time = results['fast_times'][i]

            if fast_time > 0:
                ratio = manual_time / fast_time
                plt.annotate(f'{ratio:.1f}x',
                             xy=(N, np.sqrt(manual_time * fast_time) if manual_time * fast_time > 0 else manual_time),
                             xytext=(5, 0),
                             textcoords='offset points',
                             fontsize=8)

        plt.tight_layout()
        save_path1 = os.path.join(self.output_dir, "execution_time_comparison.png")
        plt.savefig(save_path1)
        plt.show()

        # Grafico: rapporto di velocità
        plt.figure(figsize=(10, 6))

        speedup_ratios = []
        for m, f in zip(results['manual_times'], results['fast_times']):
            if f > 0:
                speedup_ratios.append(m / f)
            else:
                speedup_ratios.append(1000)

        valid_sizes = [s for i, s in enumerate(results['sizes']) if results['fast_times'][i] > 0]
        valid_ratios = [r for i, r in enumerate(speedup_ratios) if results['fast_times'][i] > 0]

        if valid_sizes and valid_ratios:
            plt.plot(valid_sizes, valid_ratios, 'o-')
            plt.xlabel('Dimensione matrice (N)')
            plt.ylabel('Rapporto di velocità (Manuale / Veloce)')
            plt.title('Rapporto di velocità tra implementazioni DCT2')
            plt.grid(True)
            plt.tight_layout()
            save_path2 = os.path.join(self.output_dir, "speedup_ratio.png")
            plt.savefig(save_path2)
            plt.show()
        else:
            print("Impossibile generare il grafico dei rapporti di velocità - dati insufficienti")


if __name__ == "__main__":
    """
        Esecuzione principale:
        - Verifica la correttezza della DCT2 con i test interni.
        - Definisce una lista di dimensioni per il benchmark.
        - Esegue il benchmark e visualizza i risultati.
    """
    transformer = DCT2()
    main = Main()

    print("Verifica DCT2:")
    transformer.verify_dct()

    sizes = [8, 16, 32, 64, 128, 256]
    print("-------------------\nDimensioni delle matrici da testare:")
    print(sizes)

    print("-------------------\nEsecuzione benchmark:")
    results = main.benchmark(sizes)

    main.plot_results(results)
