import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


class ImageUtils:
    """
        Classe di utilità per la gestione e la visualizzazione di immagini
        in interfacce grafiche Tkinter e per il salvataggio di confronti visivi.
    """
    def __init__(self):
        pass

    def display_image(self, image, canvas):
        """
            Visualizza un'immagine ridimensionata all'interno di un canvas Tkinter.

            Parametri:
                image (numpy.ndarray): L'immagine da visualizzare.
                canvas (tk.Canvas): Il canvas su cui disegnare l'immagine.
        """
        height, width = image.shape
        max_size = 280

        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)

        # Ridimensiona usando PIL
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Converti per Tkinter
        photo = ImageTk.PhotoImage(pil_image)

        # Pulisci canvas e mostra immagine
        canvas.delete("all")
        canvas.create_image(150, 150, image=photo, anchor=tk.CENTER)

        # Mantieni riferimento per evitare garbage collection
        canvas.image = photo

    def save_comparison(self, original, compressed, filepath):
        """
            Salva un confronto affiancato tra l'immagine originale e quella compressa.

            Parametri:
                original (numpy.ndarray): L'immagine originale.
                compressed (numpy.ndarray): L'immagine compressa.
                filepath (str): Il percorso completo dove salvare l'immagine di confronto.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Immagine originale
        ax1.imshow(original, cmap='gray', vmin=0, vmax=255)
        ax1.set_title('Immagine Originale')
        ax1.axis('off')

        # Immagine compressa
        ax2.imshow(compressed, cmap='gray', vmin=0, vmax=255)
        ax2.set_title('Immagine Compressa')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()