import numpy as np
from scipy.fftpack import dct, idct

def dct2(block):
    """
        Applica la trasformata discreta del coseno 2D (DCT2) a un blocco di immagine.

        La trasformata è calcolata lungo le righe e successivamente lungo le colonne,
        utilizzando la normalizzazione ortogonale.

        Args:
            block (np.ndarray): Blocco bidimensionale (F×F) di immagine.

        Returns:
            np.ndarray: Blocco trasformato nel dominio della frequenza.
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """
        Applica l'inversa della trasformata discreta del coseno 2D (IDCT2) a un blocco.

        L'inversa viene applicata riga per riga e poi colonna per colonna,
        mantenendo la normalizzazione ortogonale.

        Args:
            block (np.ndarray): Blocco trasformato nel dominio della frequenza.

        Returns:
            np.ndarray: Blocco ricostruito nel dominio spaziale.
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def apply_cutoff(dct_block, d):
    """
        Applica un cutoff in frequenza al blocco DCT2, azzerando le alte frequenze.

        Mantiene solo i coefficienti per cui k + l < d, dove k e l sono gli indici
        delle righe e colonne. Gli altri coefficienti vengono impostati a 0.

        Args:
            dct_block (np.ndarray): Blocco DCT2 su cui applicare il cutoff.
            d (int): Soglia di cutoff in frequenza.

        Returns:
            np.ndarray: Blocco DCT2 troncato.
    """
    F = dct_block.shape[0]
    for k in range(F):
        for l in range(F):
            if k + l >= d:
                dct_block[k, l] = 0
    return dct_block

def normalize_block(block):
    """
        Arrotonda e limita i valori di un blocco tra 0 e 255, convertendoli in uint8.

        Questo step è necessario per riportare i valori all'intervallo valido per
        immagini in scala di grigi a 8 bit.

        Args:
            block (np.ndarray): Blocco di immagine da normalizzare.

        Returns:
            np.ndarray: Blocco normalizzato con tipo uint8.
    """
    block = np.rint(block)
    block[block < 0] = 0
    block[block > 255] = 255
    return block.astype(np.uint8)

def compress_image(img_array, F, d):
    """
        Comprimi un'immagine in scala di grigi utilizzando la DCT2 a blocchi.

        L'immagine viene suddivisa in blocchi F×F, compressa mantenendo solo i coefficienti
        a bassa frequenza secondo una soglia d, e poi ricostruita.

        Args:
            img_array (np.ndarray): Immagine in scala di grigi (2D).
            F (int): Dimensione del blocco (es. 8 o 16).
            d (int): Cutoff di frequenza; più basso significa maggiore compressione.

        Returns:
            np.ndarray: Immagine compressa ricostruita.
    """
    h, w = img_array.shape
    h_crop = (h // F) * F
    w_crop = (w // F) * F
    img_array = img_array[:h_crop, :w_crop]
    compressed = np.zeros_like(img_array)

    for i in range(0, h_crop, F):
        for j in range(0, w_crop, F):
            block = img_array[i:i+F, j:j+F]
            c = dct2(block)
            c = apply_cutoff(c, d)
            ff = idct2(c)
            compressed[i:i+F, j:j+F] = normalize_block(ff)

    return compressed
