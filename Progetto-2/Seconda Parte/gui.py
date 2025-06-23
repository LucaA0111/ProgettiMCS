import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import os
from datetime import datetime
from utils.dct_utils import DCTProcessor
from utils.image_utils import ImageUtils
from utils.utils import Utils


class DCTImageCompressor:
    """
        Interfaccia grafica per la compressione di immagini BMP in scala di grigi
        utilizzando la Trasformata Discreta del Coseno (DCT).
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Compressore Immagini DCT")
        self.root.geometry(Utils.WINDOW_SIZE)

        # Variabili per memorizzare le immagini
        self.original_image = None
        self.compressed_image = None
        self.image_path = None

        self.dct_processor = DCTProcessor()

        self.image_utils = ImageUtils()

        self.setup_ui()

    def setup_ui(self):
        """
            Configura e costruisce l'interfaccia grafica dell'applicazione.

            Crea:
            - Frame per i controlli.
            - Input per i parametri F e d.
            - Pulsanti per selezionare l'immagine e avviare la compressione.
            - Barra di progresso.
            - Pannello per le informazioni.
            - Canvas per visualizzare immagine originale e immagine compressa.
        """

        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configurazione grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        control_frame = ttk.LabelFrame(main_frame, text="Controlli", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))

        # Selezione file
        ttk.Button(control_frame, text="Seleziona Immagine BMP",
                   command=self.select_image).grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W + tk.E)

        # Parametro F (dimensione blocco)
        ttk.Label(control_frame, text="F (Dimensione blocco):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.f_var = tk.StringVar(value=Utils.DEFAULT_F)
        f_spinbox = ttk.Spinbox(control_frame, from_=4, to=32, width=10, textvariable=self.f_var)
        f_spinbox.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Parametro d (soglia taglio frequenze)
        ttk.Label(control_frame, text="d (Soglia taglio):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.d_var = tk.StringVar(value=Utils.DEFAULT_D)
        self.d_spinbox = ttk.Spinbox(control_frame, from_=0, to=62, width=10, textvariable=self.d_var)
        self.d_spinbox.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Aggiorna il limite di d quando F cambia
        self.f_var.trace('w', self.update_d_limit)

        # Pulsante compressione
        ttk.Button(control_frame, text="Comprimi Immagine",
                   command=self.compress_image).grid(row=3, column=0, columnspan=2, pady=10, sticky=tk.W + tk.E)

        # Barra di progresso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        # Informazioni
        self.info_text = tk.Text(control_frame, height=8, width=40)
        self.info_text.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

        # Scrollbar per il testo
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=5, column=2, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=scrollbar.set)

        # Frame per le immagini
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)

        # Canvas per immagine originale
        self.original_canvas = tk.Canvas(image_frame, width=300, height=300, bg="white")
        self.original_canvas.grid(row=1, column=0, padx=5, pady=5)

        # Canvas per immagine compressa
        self.compressed_canvas = tk.Canvas(image_frame, width=300, height=300, bg="white")
        self.compressed_canvas.grid(row=1, column=1, padx=5, pady=5)

        # Etichette
        ttk.Label(image_frame, text="Immagine Originale").grid(row=0, column=0, pady=5)
        ttk.Label(image_frame, text="Immagine Compressa").grid(row=0, column=1, pady=5)

    def update_d_limit(self, *args):
        """
            Aggiorna dinamicamente il limite massimo selezionabile per la soglia d
            in base al valore corrente di F.

            Impedisce che il valore di d superi 2*F - 2.
        """
        try:
            F = int(self.f_var.get())
            max_d = 2 * F - 2
            self.d_spinbox.configure(to=max_d)
            if int(self.d_var.get()) > max_d:
                self.d_var.set(str(max_d))
        except ValueError:
            pass

    def select_image(self):
        """
            Apre una finestra di dialogo per selezionare un file BMP.

            Carica l'immagine selezionata in scala di grigi e la visualizza
            nel canvas dedicato all'immagine originale.

            Mostra inoltre informazioni sul file caricato.
        """
        default_path = Utils.IMAGE_DIR

        if not os.path.exists(default_path):
            default_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        file_path = filedialog.askopenfilename(
            title="Seleziona immagine BMP",
            initialdir=default_path,
            filetypes=[("BMP files", "*.bmp"), ("All files", "*.*")]
        )

        if file_path:
            try:
                pil_image = Image.open(file_path).convert('L')
                self.original_image = np.array(pil_image)
                self.image_path = file_path

                if self.original_image is None:
                    raise ValueError("Impossibile caricare l'immagine")

                self.image_utils.display_image(self.original_image, self.original_canvas)

                filename = os.path.basename(file_path)
                self.update_info(f"Immagine caricata: {filename}\n")
                self.update_info(f"Dimensioni: {self.original_image.shape[1]}x{self.original_image.shape[0]}\n")

            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel caricamento dell'immagine: {str(e)}")

    def update_info(self, text):
        """
            Aggiorna il pannello informazioni aggiungendo un nuovo messaggio.
        """
        self.info_text.insert(tk.END, text)
        self.info_text.see(tk.END)
        self.info_text.update()

    def update_progress(self, value):
        """
            Aggiorna la barra di progresso con il valore corrente.
        """
        self.progress_var.set(value)
        self.root.update_idletasks()

    def compress_image(self):
        """
            Esegue la compressione DCT dell'immagine caricata.

            Controlla che un'immagine sia stata selezionata.
            Applica la compressione utilizzando il DCTProcessor.
            Aggiorna la barra di progresso durante l'elaborazione.
            Visualizza l'immagine compressa e le statistiche di compressione.
            Salva automaticamente l'immagine di confronto.
        """
        if self.original_image is None:
            messagebox.showwarning("Attenzione", "Seleziona prima un'immagine")
            return

        try:
            F = int(self.f_var.get())
            d = int(self.d_var.get())

            self.update_info(f"\nInizio compressione con F={F}, d={d}\n")
            self.update_progress(0)

            # Comprimi l'immagine
            self.compressed_image, stats = self.dct_processor.compress_image(
                self.original_image, F, d, self.update_progress
            )

            # Mostra l'immagine compressa
            self.image_utils.display_image(self.compressed_image, self.compressed_canvas)

            # Mostra statistiche
            self.update_info(f"\nCompressione completata!\n")
            self.update_info(f"MSE: {stats['mse']:.2f}\n")
            self.update_info(f"PSNR: {stats['psnr']:.2f} dB\n")
            self.update_info(
                f"Coefficienti mantenuti: {stats['kept_coeffs']}/{stats['total_coeffs']} ({stats['compression_ratio']:.2%})\n")

            # Salva il confronto
            self.save_comparison()
            self.update_progress(100)

        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la compressione: {str(e)}")
            self.update_info(f"Errore: {str(e)}\n")

    def save_comparison(self):
        """
            Salva un'immagine di confronto tra l'originale e la versione compressa.

            L'immagine viene salvata nella directory definita in Utils.OUTPUT_DIR
            con un nome che include un timestamp per evitare sovrascritture.
        """
        try:
            # Crea la cartella se non esiste
            output_dir = Utils.OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)

            # Nome file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)

            # Crea l'immagine di confronto
            self.image_utils.save_comparison(self.original_image, self.compressed_image, filepath)

            self.update_info(f"Confronto salvato in: compressed_images/{filename}\n")

        except Exception as e:
            self.update_info(f"Errore nel salvataggio: {str(e)}\n")