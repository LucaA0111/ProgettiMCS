import numpy as np
from scipy.fft import dctn, idctn


class DCTProcessor:
    """
        Classe per la compressione di immagini in scala di grigi tramite la Trasformata Coseno Discreta (DCT).

        Questa classe consente di applicare la DCT bidimensionale (DCT2) a blocchi di un'immagine,
        comprimendo l'immagine attraverso l'eliminazione di componenti ad alta frequenza.
    """
    def __init__(self):
        pass

    def dct2_library(self, block):
        """
            Applica la DCT bidimensionale (DCT2) a un blocco usando scipy.

            Parametri:
                block (numpy.ndarray): Il blocco di immagine da trasformare.

            Ritorna:
                numpy.ndarray: Il blocco trasformato nel dominio delle frequenze.
        """
        return dctn(block, type=2, norm='ortho')

    def idct2_library(self, block):
        """
            Applica l'inversa della DCT bidimensionale (IDCT2) a un blocco usando scipy.

            Parametri:
                block (numpy.ndarray): Il blocco nel dominio delle frequenze da riconvertire.

            Ritorna:
                numpy.ndarray: Il blocco ricostruito nel dominio spaziale.
        """
        return idctn(block, type=2, norm='ortho')

    def compress_block(self, block, d):
        """
            Comprime un singolo blocco di immagine applicando la DCT e azzerando
            le frequenze con indice k + l >= d.

            Parametri:
                block (numpy.ndarray): Il blocco di immagine da comprimere.
                d (int): Soglia per la ritenzione dei coefficienti DCT (k + l < d).

            Ritorna:
                numpy.ndarray: Il blocco compresso e ricostruito.
        """
        F = block.shape[0]

        # Applica DCT2
        dct_coeffs = self.dct2_library(block.astype(np.float64))

        # Rimuovo frequenze con k + l >= d
        for k in range(F):
            for l in range(F):
                if k + l >= d:
                    dct_coeffs[k, l] = 0

        # Applica IDCT2
        reconstructed = self.idct2_library(dct_coeffs)

        reconstructed = np.round(reconstructed)
        reconstructed = np.clip(reconstructed, 0, 255)

        return reconstructed.astype(np.uint8)

    def compress_image(self, image, F, d, progress_callback=None):
        """
            Comprime l'intera immagine suddividendola in blocchi di dimensione F x F
            e applicando la compressione DCT su ciascun blocco.

            Parametri:
                image (numpy.ndarray): L'immagine da comprimere (grayscale).
                F (int): Dimensione dei blocchi quadrati.
                d (int): Soglia di compressione (frequenze con k + l >= d saranno eliminate).
                progress_callback (funzione, opzionale): Funzione per aggiornare la barra di progresso.

            Ritorna:
                tuple:
                    - numpy.ndarray: L'immagine compressa.
                    - dict: Statistiche di compressione (MSE, PSNR, rapporto di compressione).
        """
        height, width = image.shape

        blocks_h = height // F
        blocks_w = width // F

        # Crea l'immagine compressa
        compressed_height = blocks_h * F
        compressed_width = blocks_w * F
        compressed_image = np.zeros((compressed_height, compressed_width), dtype=np.uint8)

        # Processo ogni blocco
        total_blocks = blocks_h * blocks_w

        for i in range(blocks_h):
            for j in range(blocks_w):
                start_row = i * F
                end_row = start_row + F
                start_col = j * F
                end_col = start_col + F

                block = image[start_row:end_row, start_col:end_col]

                compressed_block = self.compress_block(block, d)

                compressed_image[start_row:end_row, start_col:end_col] = compressed_block

                # Aggiorna progresso
                if progress_callback:
                    progress = ((i * blocks_w + j + 1) / total_blocks) * 100
                    progress_callback(progress)

        # Calcola statistiche
        stats = self.calculate_stats(image[:compressed_height, :compressed_width], compressed_image, F, d)

        return compressed_image, stats

    def calculate_stats(self, original, compressed, F, d):
        """
            Calcola le statistiche di compressione tra l'immagine originale e quella compressa.

            Parametri:
                original (numpy.ndarray): L'immagine originale.
                compressed (numpy.ndarray): L'immagine compressa.
                F (int): Dimensione dei blocchi.
                d (int): Soglia di compressione (frequenze con k + l >= d sono eliminate).

            Ritorna:
                dict: Dizionario contenente:
                    - mse: Errore quadratico medio (Mean Squared Error).
                    - psnr: Rapporto segnale-rumore di picco (Peak Signal-to-Noise Ratio).
                    - kept_coeffs: Numero di coefficienti DCT mantenuti.
                    - total_coeffs: Numero totale di coefficienti per blocco.
                    - compression_ratio: Percentuale di coefficienti mantenuti.
        """
        mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')

        # Calcola il rapporto di compressione
        total_coeffs = F * F
        kept_coeffs = 0
        for k in range(F):
            for l in range(F):
                if k + l < d:
                    kept_coeffs += 1

        compression_ratio = kept_coeffs / total_coeffs

        return {
            'mse': mse,
            'psnr': psnr,
            'kept_coeffs': kept_coeffs,
            'total_coeffs': total_coeffs,
            'compression_ratio': compression_ratio
        }