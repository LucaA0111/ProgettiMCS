import matplotlib.pyplot as plt
import os
from datetime import datetime

def show_images(original, compressed, F, d):
    """
        Visualizza affiancate l'immagine originale e quella compressa,
        e salva il risultato con un nome basato su timestamp.

        La funzione crea una figura con due sottotrame:
        - a sinistra l'immagine originale
        - a destra l'immagine compressa con indicazione dei parametri F e d.

        L'immagine risultante viene salvata in una cartella chiamata `compressed_images`
        nello stesso percorso del file Python che esegue la funzione.

        Args:
            original (np.ndarray): Immagine originale in scala di grigi (2D).
            compressed (np.ndarray): Immagine compressa da confrontare (2D).
            F (int): Dimensione dei blocchi usati per la compressione (es. 8 o 16).
            d (int): Soglia di cutoff per la compressione (più basso = più compressione).

        Returns:
            None
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Originale')
    axs[0].axis('off')

    axs[1].imshow(compressed, cmap='gray')
    axs[1].set_title(f'Compressa (F={F}, d={d})')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


    save_dir = os.path.join(os.path.dirname(__file__), "compressed_images")
    os.makedirs(save_dir, exist_ok=True)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath)
