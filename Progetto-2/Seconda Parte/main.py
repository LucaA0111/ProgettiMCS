import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np
from dct_utils import compress_image
from image_utils import show_images

class App(tk.Tk):
    """
        GUI per la compressione di immagini BMP con DCT2.

        Funzionalità:
        - Selezione file immagine BMP in scala di grigi
        - Inserimento dei parametri:
            F: dimensione del blocco per la DCT (es. 8, 16)
            d: soglia di cutoff per l'eliminazione delle frequenze (0 <= d <= 2F - 2)
        - Compressione dell'immagine tramite DCT2
        - Visualizzazione dell'immagine originale e compressa
        - Salvataggio automatico del confronto

        Attributi:
            initial_dir (str): Directory predefinita per il file dialog.
            path_var (tk.StringVar): Percorso dell'immagine selezionata.
            f_var (tk.IntVar): Dimensione dei blocchi F.
            d_var (tk.IntVar): Soglia di cutoff d.
    """
    def __init__(self):
        super().__init__()
        self.title("DCT Image Compressor")
        self.geometry("600x180")
        self.resizable(True, True)

        self.initial_dir = os.path.join(os.path.dirname(__file__), "immagini")
        os.makedirs(self.initial_dir, exist_ok=True)

        # Path immagine
        tk.Label(self, text="Immagine BMP:").grid(row=0, column=0, pady=10, sticky="e")
        browse_btn = tk.Button(self, text="Sfoglia...", command=self.browse_file)
        browse_btn.grid(row=0, column=1, padx=5, sticky="w")
        self.path_var = tk.StringVar()
        entry = tk.Entry(self, textvariable=self.path_var, width=40)
        entry.grid(row=0, column=2, padx=5, sticky="w")

        # Parametro F
        tk.Label(self, text="Blocchi F:").grid(row=1, column=0, sticky="e")
        self.f_var = tk.IntVar(value=8)
        tk.Entry(self, textvariable=self.f_var, width=5).grid(row=1, column=1, sticky="w")

        # Parametro d
        tk.Label(self, text="Soglia d:").grid(row=2, column=0, sticky="e")
        self.d_var = tk.IntVar(value=10)
        tk.Entry(self, textvariable=self.d_var, width=5).grid(row=2, column=1, sticky="w")

        # Pulsante esecuzione
        tk.Button(self, text="Comprimi", command=self.run).grid(row=3, column=1, pady=20)

    def browse_file(self):
        """
            Apre un file dialog per selezionare un'immagine BMP.
            Aggiorna `self.path_var` con il percorso selezionato.
        """
        path = filedialog.askopenfilename(
            initialdir=self.initial_dir,
            title="Seleziona immagine BMP",
            filetypes=[("Bitmap files", "*.bmp")]
        )
        if path:
            self.path_var.set(path)

    def run(self):
        """
            Esegue il processo di compressione quando l'utente preme "Comprimi".
            Include:
            - Controllo dei parametri
            - Caricamento immagine
            - Compressione tramite `compress_image`
            - Visualizzazione con `show_images`
        """
        path = self.path_var.get()
        F = self.f_var.get()
        d = self.d_var.get()

        if not os.path.isfile(path):
            messagebox.showerror("Errore", "Seleziona un file valido.")
            return
        if not (1 <= F and 0 <= d <= 2*F-2):
            messagebox.showerror("Errore", "Valori F o d non validi.")
            return

        img = Image.open(path).convert('L')
        img_array = np.array(img)
        compressed = compress_image(img_array, F, d)
        show_images(img_array, compressed, F, d)

if __name__ == "__main__":
    App().mainloop()
