import os

class Utils:
    """
        Classe di utilità per la gestione dei percorsi e delle configurazioni del progetto.

        Attributi di classe:
            PROJECT_ROOT (str): Percorso assoluto della root del progetto,
                calcolato risalendo di un livello rispetto alla directory del file corrente.

            IMAGE_DIR (str): Percorso assoluto alla directory contenente le immagini originali.

            OUTPUT_DIR (str): Percorso assoluto alla directory destinata a contenere
                le immagini compresse o elaborate.

            DEFAULT_F (int): Valore di default per il parametro F. L'uso specifico
                dipende dal contesto applicativo (es. fattore di compressione).

            DEFAULT_D (int): Valore di default per il parametro D. L'uso specifico
                dipende dal contesto applicativo (es. profondità, distanza, ecc.).

            WINDOW_SIZE (str): Dimensione predefinita della finestra grafica in pixel,
                espressa nel formato "larghezza x altezza" (es. "1200x800").
        """

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'compressed_images')

    DEFAULT_F = 8
    DEFAULT_D = 10

    WINDOW_SIZE = "1200x800"