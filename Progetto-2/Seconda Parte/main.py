from tkinter import Tk, filedialog, simpledialog
from PIL import Image
import numpy as np
import os
from image_utils import show_images
from dct_utils import compress_image

def main():
    Tk().withdraw()

    # Percorso predefinito: cartella "immagini" accanto a main.py
    default_dir = os.path.join(os.path.dirname(__file__), "immagini")

    path = filedialog.askopenfilename(
        title="Scegli un'immagine BMP in toni di grigio",
        initialdir=default_dir,
        filetypes=[("Bitmap files", "*.bmp")]
    )
    if not path:
        return

    img = Image.open(path).convert('L')
    img_array = np.array(img)

    F = simpledialog.askinteger("Input", "Ampiezza blocchi F (es: 8, 16, ...):", minvalue=1)
    d = simpledialog.askinteger("Input", f"Soglia frequenze d (0 - {2 * F - 2}):", minvalue=0, maxvalue=2 * F - 2)

    compressed = compress_image(img_array, F, d)

    show_images(img_array, compressed, F, d)

if __name__ == "__main__":
    main()
