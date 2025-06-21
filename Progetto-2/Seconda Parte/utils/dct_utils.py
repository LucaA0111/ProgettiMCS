import numpy as np
from scipy.fft import dctn, idctn


class DCTProcessor:
    def __init__(self):
        pass

    def dct2_library(self, block):
        """DCT2 usando la libreria scipy"""
        return dctn(block, type=2, norm='ortho')

    def idct2_library(self, block):
        """IDCT2 usando la libreria scipy"""
        return idctn(block, type=2, norm='ortho')

    def compress_block(self, block, d):
        """Comprime un singolo blocco usando DCT"""
        F = block.shape[0]

        # Applica DCT2
        dct_coeffs = self.dct2_library(block.astype(np.float64))

        # Elimina le frequenze con k + l >= d
        for k in range(F):
            for l in range(F):
                if k + l >= d:
                    dct_coeffs[k, l] = 0

        # Applica IDCT2
        reconstructed = self.idct2_library(dct_coeffs)

        # Arrotonda e limita i valori
        reconstructed = np.round(reconstructed)
        reconstructed = np.clip(reconstructed, 0, 255)

        return reconstructed.astype(np.uint8)

    def compress_image(self, image, F, d, progress_callback=None):
        """Comprime l'intera immagine"""
        height, width = image.shape

        # Calcola quanti blocchi completi possiamo ottenere
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
                # Estrai il blocco
                start_row = i * F
                end_row = start_row + F
                start_col = j * F
                end_col = start_col + F

                block = image[start_row:end_row, start_col:end_col]

                # Comprimi il blocco
                compressed_block = self.compress_block(block, d)

                # Inserisci il blocco compresso nell'immagine risultante
                compressed_image[start_row:end_row, start_col:end_col] = compressed_block

                # Aggiorna progresso
                if progress_callback:
                    progress = ((i * blocks_w + j + 1) / total_blocks) * 100
                    progress_callback(progress)

        # Calcola statistiche
        stats = self.calculate_stats(image[:compressed_height, :compressed_width], compressed_image, F, d)

        return compressed_image, stats

    def calculate_stats(self, original, compressed, F, d):
        """Calcola le statistiche di compressione"""
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