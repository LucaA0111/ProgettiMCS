import tkinter as tk
from gui import DCTImageCompressor


def main():
    """
        Funzione principale che avvia l'applicazione Tkinter.

        Crea una finestra principale, inizializza l'applicazione DCTImageCompressor
        e avvia il loop principale dell'interfaccia grafica.
    """
    root = tk.Tk()
    app = DCTImageCompressor(root)
    root.mainloop()


if __name__ == "__main__":
    """
        Punto di ingresso dell'applicazione.

        Quando il file viene eseguito direttamente, chiama la funzione main()
        per avviare l'interfaccia grafica.
    """
    main()